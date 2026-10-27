"""Educational EOD trading lab. No broker orders or profitability claims."""
from __future__ import annotations
import argparse
import warnings
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'ergebnisse'
OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']


def normalize_ohlcv(raw: pd.DataFrame, ticker: str, min_rows: int = 160,
                    bad_row_policy: str = 'drop', max_bad_fraction: float = 0.02) -> pd.DataFrame:
    """Audit suspicious candles; reject recent, excessive, or ambiguous defects.

    'drop' removes a *small* number of old invalid rows and records the audit.
    It never interpolates prices; 'strict' rejects any invalid OHLCV bar.
    """
    ticker = ticker.strip().upper()
    if not ticker or raw is None or raw.empty:
        raise ValueError('Ticker oder Kursdaten fehlen')
    if bad_row_policy not in ('drop', 'strict') or not (0 <= max_bad_fraction < 1):
        raise ValueError('Ungueltige Datenqualitaets-Parameter')
    if isinstance(raw.columns, pd.MultiIndex):
        matches = [i for i in range(raw.columns.nlevels)
                   if ticker in raw.columns.get_level_values(i)]
        if len(matches) != 1:
            raise ValueError(f'MultiIndex nicht eindeutig fuer {ticker}')
        raw = raw.xs(ticker, axis=1, level=matches[0], drop_level=True)
    if isinstance(raw.columns, pd.MultiIndex) or not raw.columns.is_unique:
        raise ValueError('Uneindeutige OHLCV-Spalten')
    if not set(OHLCV).issubset(raw.columns):
        raise ValueError('OHLCV-Spalten fehlen')
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError('Datumsindex fehlt')
    duplicates = int(raw.index.duplicated(keep='last').sum())
    df = raw.loc[~raw.index.duplicated(keep='last'), OHLCV].sort_index()
    df = df.apply(pd.to_numeric, errors='coerce')
    if df.empty:
        raise ValueError('Keine Kursdaten nach Datumskontrolle')
    price_cols = OHLCV[:4]
    malformed = df[price_cols].isna().any(axis=1)
    malformed |= (df[price_cols] <= 0).any(axis=1)
    malformed |= df['High'] < df[['Open', 'Close', 'Low']].max(axis=1)
    malformed |= df['Low'] > df[['Open', 'Close', 'High']].min(axis=1)
    malformed |= df['Volume'].notna() & (df['Volume'] < 0)
    bad = int(malformed.sum())
    if bad and bool(malformed.iloc[-1]):
        raise ValueError('Letzter Tagesbalken unplausibel: keinen alten Kurs vortaeuschen')
    if bad and bad_row_policy == 'strict':
        raise ValueError(f'Unplausible OHLCV-Werte in {bad} Zeilen')
    if bad > max_bad_fraction * len(df):
        raise ValueError(f'Zu viele unplausible OHLCV-Werte: {bad}/{len(df)}')
    df = df.loc[~malformed].copy()
    if len(df) < min_rows:
        raise ValueError(f'Weniger als {min_rows} verwertbare Handelstage')
    report = {'Ticker': ticker, 'Eingangszeilen': len(raw),
              'Doppelte_Daten': duplicates, 'Entfernte_Balken': bad,
              'Verwertbare_Zeilen': len(df), 'Policy': bad_row_policy}
    df.attrs['quality_report'] = report
    if bad or duplicates:
        warnings.warn(f'Datenqualitaet {ticker}: {duplicates} Dubletten, '
                      f'{bad} unplausible historische Balken entfernt; '
                      'Indikatoren nach Datenbereinigung neu berechnen.',
                      UserWarning, stacklevel=2)
    return df

def fetch(ticker: str, period: str = '5y') -> pd.DataFrame:
    """Live data dependency is imported only when actually fetching."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError('yfinance fehlt: pip install -r requirements.txt') from exc
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError('Ticker ist leer')
    raw = yf.download(ticker, period=period, interval='1d', auto_adjust=True,
                      progress=False, threads=False, group_by='ticker')
    return normalize_ohlcv(raw, ticker)


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df['Close']
    df['SMA20'] = c.rolling(20).mean()
    df['SMA50'] = c.rolling(50).mean()
    diff = c.diff()
    gains = diff.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    losses = (-diff.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gains / losses.replace(0, np.nan)
    df['RSI14'] = 100 - 100/(1+rs)
    df.loc[(losses == 0) & (gains > 0), 'RSI14'] = 100
    df.loc[(losses == 0) & (gains == 0), 'RSI14'] = 50
    prev = c.shift(1)
    tr = pd.concat([df['High']-df['Low'], (df['High']-prev).abs(),
                    (df['Low']-prev).abs()], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    df['Signal'] = ((c > df['SMA50']) & (df['SMA20'] > df['SMA50']) &
                    df['RSI14'].between(50, 70))
    return df


def backtest(df: pd.DataFrame, fee: float = 0.001, stop_atr: float = 2.,
             target_atr: float = 3., max_days: int = 10,
             slippage: float = 0.) -> pd.DataFrame:
    """Close signal, next-open entry, explicit overnight gap; OHLC fill assumptions."""
    if not (0 <= fee < 1 and 0 <= slippage < 1) or stop_atr <= 0 or target_atr <= 0 or max_days < 1:
        raise ValueError('Ungueltige Backtest-Parameter')
    required = {'Open', 'High', 'Low', 'Close', 'Signal', 'ATR14'}
    if not required.issubset(df.columns):
        raise ValueError('Backtest-Spalten fehlen')
    trades = []
    i = 60
    while i < len(df)-1:
        if not bool(df['Signal'].iloc[i]) or pd.isna(df['ATR14'].iloc[i]):
            i += 1
            continue
        entry_i = i+1
        entry = float(df['Open'].iloc[entry_i])
        signal_close = float(df['Close'].iloc[i])
        atr = float(df['ATR14'].iloc[i])
        if atr <= 0 or entry <= 0 or signal_close <= 0:
            i += 1
            continue
        stop, target = entry-stop_atr*atr, entry+target_atr*atr
        if stop <= 0:
            i += 1
            continue
        last_i = min(entry_i+max_days-1, len(df)-1)
        exit_i, exit_price, reason = last_i, float(df['Close'].iloc[last_i]), 'Zeitlimit: letzter Close'
        for j in range(entry_i, last_i+1):
            o, h, low = (float(df[col].iloc[j]) for col in ['Open', 'High', 'Low'])
            if o <= stop:
                exit_i, exit_price, reason = j, o, 'Gap unter Stop'
            elif o >= target:
                exit_i, exit_price, reason = j, o, 'Gap ueber Ziel'
            elif low <= stop:
                exit_i, exit_price, reason = j, stop, 'Stop intraday (konservativ)'
            elif h >= target:
                exit_i, exit_price, reason = j, target, 'Ziel intraday'
            else:
                continue
            break
        gross_buy = entry*(1+slippage)*(1+fee)
        net_sell = exit_price*(1-slippage)*(1-fee)
        net = net_sell/gross_buy-1
        trades.append({'Signal': str(df.index[i].date()),
                       'Einstieg': str(df.index[entry_i].date()),
                       'Ausstieg': str(df.index[exit_i].date()),
                       'Signal_Schlusskurs': signal_close,
                       'Ausfuehrungsgap_pct': 100*(entry/signal_close-1),
                       'Einstiegspreis': entry, 'Stop': stop, 'Ziel': target,
                       'Ausstiegspreis': exit_price, 'Grund': reason,
                       'Gebuehrensatz': fee, 'Slippage': slippage,
                       'Kaufpreis_inkl_Kosten': gross_buy,
                       'Verkaufspreis_nach_Kosten': net_sell,
                       'Rendite_nach_Modellkosten': net})
        i = exit_i+1
    columns = ['Signal', 'Einstieg', 'Ausstieg', 'Signal_Schlusskurs',
               'Ausfuehrungsgap_pct', 'Einstiegspreis', 'Stop', 'Ziel',
               'Ausstiegspreis', 'Grund', 'Gebuehrensatz', 'Slippage',
               'Kaufpreis_inkl_Kosten', 'Verkaufspreis_nach_Kosten',
               'Rendite_nach_Modellkosten']
    return pd.DataFrame(trades, columns=columns)

def equity_from_trades(trades: pd.DataFrame, start_cash: float = 10000.,
                       risk_fraction: float = 0.01) -> pd.DataFrame:
    """Cash-only, sequential closed-trade ledger with fee-aware affordability.

    Risk uses a modeled stop execution, not a loss guarantee: opening gaps
    and missing liquidity can produce much larger losses.
    """
    if not np.isfinite(start_cash) or start_cash <= 0 or not (0 < risk_fraction <= 1):
        raise ValueError('Ungueltige Kapitalparameter')
    cash = float(start_cash)
    rows = [{'Datum': 'Start', 'Kapital': cash, 'Stueck': 0, 'Gewinn': 0.,
             'Kaufwert_inkl_Kosten': 0., 'Restgeld_nach_Kauf': cash,
             'Geplantes_Stoprisiko': 0.}]
    for _, t in trades.iterrows():
        entry, stop = float(t['Einstiegspreis']), float(t['Stop'])
        fee = float(t.get('Gebuehrensatz', 0.))
        slip = float(t.get('Slippage', 0.))
        if not (np.isfinite(entry) and np.isfinite(stop) and 0 < stop < entry
                and 0 <= fee < 1 and 0 <= slip < 1):
            raise ValueError('Ungueltiger Trade fuer Kapitalrechnung')
        gross_buy = entry*(1+slip)*(1+fee)
        stop_net_sell = stop*(1-slip)*(1-fee)
        stop_risk = gross_buy-stop_net_sell
        max_risk_qty = math.floor(cash*risk_fraction/stop_risk)
        max_cash_qty = math.floor(cash/gross_buy)
        qty = max(0, min(max_risk_qty, max_cash_qty))
        if 'Verkaufspreis_nach_Kosten' in t and pd.notna(t['Verkaufspreis_nach_Kosten']):
            net_sell = float(t['Verkaufspreis_nach_Kosten'])
        else:
            net_sell = float(t['Ausstiegspreis'])*(1-slip)*(1-fee)
        if not np.isfinite(net_sell) or net_sell < 0:
            raise ValueError('Ungueltiger Verkaufserloes')
        invested = qty*gross_buy
        remaining = cash-invested
        if remaining < -1e-8:
            raise AssertionError('Kapitalmodell kauft ueber verfuegbares Geld')
        profit = qty*(net_sell-gross_buy)
        cash += profit
        rows.append({'Datum': t['Ausstieg'], 'Kapital': cash, 'Stueck': qty,
                     'Gewinn': profit, 'Kaufwert_inkl_Kosten': invested,
                     'Restgeld_nach_Kauf': remaining,
                     'Geplantes_Stoprisiko': qty*stop_risk})
    return pd.DataFrame(rows)

def ml_walk_forward(df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """Four expanding out-of-sample folds + untouched final holdout.

    Each fold performs separate chronological inner tuning with purged gaps.
    These scores are descriptive for overlapping labels, not independent
    statistical samples or an investment performance validation.
    """
    if horizon < 1:
        raise ValueError('horizon muss positiv sein')
    x = pd.DataFrame(index=df.index)
    x['ret5'] = df['Close'].pct_change(5)
    x['dist20'] = df['Close']/df['SMA20'].replace(0, np.nan)-1
    x['dist50'] = df['Close']/df['SMA50'].replace(0, np.nan)-1
    x['rsi'] = df['RSI14']
    x['vol20'] = df['Close'].pct_change().rolling(20).std()
    future = df['Close'].shift(-horizon)
    x['y'] = np.where(future.notna(), (future > df['Close']).astype(int), np.nan)
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    features = ['ret5', 'dist20', 'dist50', 'rsi', 'vol20']
    cut = int(len(x)*0.8)
    develop, holdout = x.iloc[:cut], x.iloc[cut:]
    if len(develop) < 300 or len(holdout) < 40:
        raise ValueError('Zu wenige Daten fuer Walk-forward-Auswertung')

    def train_model(available: pd.DataFrame):
        if len(available) < 90 or available['y'].nunique() < 2:
            raise ValueError('Zu wenige Daten oder nur eine Klasse im Training')
        inner = TimeSeriesSplit(n_splits=3, gap=horizon)
        scores = {}
        for c in (0.01, 0.1, 1., 10.):
            fold_scores = []
            for tr_i, va_i in inner.split(available):
                tr, val = available.iloc[tr_i], available.iloc[va_i]
                if tr['y'].nunique() < 2:
                    raise ValueError('Trainingsfenster enthaelt nur eine Klasse')
                m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
                m.fit(tr[features], tr['y'].astype(int))
                fold_scores.append(balanced_accuracy_score(val['y'], m.predict(val[features])))
            scores[c] = float(np.mean(fold_scores))
        best = max(scores, key=lambda c: (scores[c], -c))
        model = make_pipeline(StandardScaler(), LogisticRegression(C=best, max_iter=2000))
        model.fit(available[features], available['y'].astype(int))
        return best, scores[best], model

    def evaluate(name: str, available: pd.DataFrame, evaluation: pd.DataFrame):
        best, cv, model = train_model(available)
        pred = model.predict(evaluation[features])
        # Training-majority baseline is frozen before evaluation starts.
        base = np.full(len(evaluation), int(available['y'].mean() >= 0.5))
        return {'Segment': name, 'Beginn': str(evaluation.index[0].date()),
                'Ende': str(evaluation.index[-1].date()),
                'C': best, 'CV_balanced_accuracy': cv,
                'Test_accuracy': accuracy_score(evaluation['y'], pred),
                'Test_balanced_accuracy': balanced_accuracy_score(evaluation['y'], pred),
                'Baseline_accuracy': accuracy_score(evaluation['y'], base),
                'Testbeobachtungen': len(evaluation),
                'Trainingsbeobachtungen': len(available)}

    # Outer expanding windows, with horizon rows purged before each test block.
    splitter = TimeSeriesSplit(n_splits=4, gap=horizon)
    rows = []
    for number, (train_i, test_i) in enumerate(splitter.split(develop), start=1):
        rows.append(evaluate(f'WF{number}', develop.iloc[train_i], develop.iloc[test_i]))
    # Last 20% is evaluated once; labels at the boundary cannot leak into training.
    rows.append(evaluate('HOLDOUT', develop.iloc[:-horizon], holdout))
    return pd.DataFrame(rows)

def scan(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            df = indicators(fetch(ticker))
            x = df.iloc[-1]
            rows.append({'Ticker': ticker, 'Datum': str(df.index[-1].date()),
                         'Close': round(float(x['Close']), 4),
                         'RSI14': round(float(x['RSI14']), 2),
                         'Signal': bool(x['Signal']), 'Status': 'OK',
                         'Entfernte_Balken': df.attrs.get('quality_report', {}).get('Entfernte_Balken', 0)})
        except Exception as exc:
            rows.append({'Ticker': ticker, 'Status': f'FEHLER: {exc}'})
    result = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    result.to_csv(OUT/'scan.csv', sep=';', index=False, encoding='utf-8-sig')
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description='Trading-Lernlabor, keine Brokerorders')
    p.add_argument('modus', choices=['scan', 'backtest', 'ml'])
    p.add_argument('--ticker', default='AAPL')
    p.add_argument('--fee', type=float, default=0.001)
    p.add_argument('--stop_atr', type=float, default=2.)
    p.add_argument('--target_atr', type=float, default=3.)
    p.add_argument('--slippage', type=float, default=0.)
    args = p.parse_args(argv)
    if not (0 <= args.fee < 1 and 0 <= args.slippage < 1) or args.stop_atr <= 0 or args.target_atr <= 0:
        p.error('Kosten muessen zwischen 0 und 1, ATR-Multiplikatoren positiv sein')
    OUT.mkdir(exist_ok=True)
    if args.modus == 'scan':
        tickers = pd.read_csv(ROOT/'aktien.csv')['Ticker'].dropna().astype(str).tolist()
        result = scan(tickers)
        print(result.to_string(index=False))
        if result.empty or not result['Status'].eq('OK').any():
            p.exit(1, 'FEHLER: Kein Ticker konnte verarbeitet werden. Pruefe Paketinstallation, Internet und Symbole.\n')
        return
    try:
        df = indicators(fetch(args.ticker))
    except (RuntimeError, ValueError, ConnectionError) as exc:
        p.exit(1, f'FEHLER beim Datenabruf: {exc}\n')
    if args.modus == 'backtest':
        trades = backtest(df, fee=args.fee, stop_atr=args.stop_atr,
                          target_atr=args.target_atr, slippage=args.slippage)
        trades.to_csv(OUT/'backtest_trades.csv', sep=';', index=False, encoding='utf-8-sig')
        equity = equity_from_trades(trades)
        equity.to_csv(OUT/'kapitalmodell.csv', sep=';', index=False, encoding='utf-8-sig')
        print(f'{len(trades)} Modelltrades. Vereinfachtes Kapitalmodell: {equity.Kapital.iloc[-1]:.2f}')
    else:
        report = ml_walk_forward(df)
        report.to_csv(OUT/'ml_report.csv', sep=';', index=False, encoding='utf-8-sig')
        print(report.to_string(index=False))


if __name__ == '__main__':
    main()
