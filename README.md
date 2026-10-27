KI-TRADING – Projektdateien zur Drei-Teile-Ausgabe

ZWEI GETRENNTE PROGRAMME. Dateien und Quellcode nicht vermischen.

A_Einsteiger/meine_trading_ki.py
  Einstieg in Python und ein statistisches ML-Modell (numpy, pandas, scikit-learn).
  Windows: Python installieren, Projektordner öffnen, dann:
  py -m venv .venv
  .venv\Scripts\python.exe -m pip install numpy pandas scikit-learn
  .venv\Scripts\python.exe meine_trading_ki.py

B_Aufbau/trading_ki.py, test_trading_ki.py, requirements.txt, aktien.csv
  Erweiterter Scanner, Backtest, zeitliche ML-Validierung und optionale Tests.
  Windows: Projektordner öffnen, dann:
  py -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  .venv\Scripts\python.exe trading_ki.py scan
  .venv\Scripts\python.exe test_trading_ki.py -v

Die Dateien des Aufbauprojekts stammen aus derselben getesteten Codebasis wie
die vorherige gedruckte Buchfassung. Der Einsteiger-Code wurde aus der
achtteiligen Programmfassung des Buchs extrahiert und syntaktisch geprüft.

WICHTIG: Dies sind Lernprogramme, keine Kaufempfehlungen, keine autonomen
Handelssysteme und kein nachgewiesener Renditevorteil.
