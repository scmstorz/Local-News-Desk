# Local News App

## Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Falls du das Environment bereits vorher installiert hattest, einmal nachziehen:

```bash
pip install -r requirements.txt --upgrade
```

Danach `local-news-app.html` direkt im Browser öffnen.

Das Frontend spricht standardmäßig mit:

`http://127.0.0.1:8765`

Ein separater Cronjob ist nicht nötig.

Der Backend-Prozess läuft dauerhaft und zieht standardmäßig alle 5 Minuten neue RSS-Einträge nach.

## Wichtige Dateien

- `local-news-app.html`
- `local_news_backend.py`
- `local_config.json`
- `LOCAL_ARCHITECTURE.md`
- `requirements.txt`

## Konfiguration

Die lokale Konfiguration liegt in:

`local_config.json`

Dort werden unter anderem gepflegt:

- Google-News-RSS-Feeds
- Ollama-Base-URL
- Ollama-Modell
- eigener Ollama-Summary-Timeout
- optionale `LLM Compare`-Modelle
- Compare-Timeout pro Modellaufruf
- Polling-Intervalle
- Server-Port

Der Python-Code liest diese Datei standardmäßig automatisch ein.
Umgebungsvariablen können die Werte weiterhin überschreiben.

## Ollama

Default-Modell:

`qwen3.5:35b`

Falls du ein anderes lokales Modell testen willst:

```bash
OLLAMA_MODEL=gemma3:27b python local_news_backend.py
```

## Lokale Daten

Standardpfade:

- SQLite: `local-news.db`
- Modelle: `models/`

Optional überschreibbar:

```bash
LOCAL_NEWS_DB_PATH=/tmp/local-news.db
LOCAL_NEWS_MODEL_DIR=/tmp/local-news-models
```

## Nützliche Umgebungsvariablen

```bash
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3.5:35b
LOCAL_NEWS_PORT=8765
LOCAL_NEWS_REFRESH_SECONDS=300
```

## Hinweise

- `claude-news-app.html` bleibt unangetastet.
- Das neue Frontend ist Desktop-first.
- Der Feed-Klassifikator ist aktuell eine transparente Baseline mit `TF-IDF + LogisticRegression`.
- Summary-Feedback wird bereits gespeichert, damit spätere Modelliteration darauf aufbauen kann.
- `trafilatura` ist optional. Wenn es lokal nicht sauber importierbar ist, fällt die Textextraktion automatisch auf `BeautifulSoup` zurück.
- `Inbox-Reset` archiviert alle aktuell offenen Feed-Einträge, ohne sie aus der Datenbank zu löschen.
- `LLM Compare` erzeugt pro aktivierter Session eine eigene Markdown-Datei in `compare_exports/`.
- Wenn ein Artikel wegen Paywall, Popup oder Blockade nicht sauber extrahiert werden kann, zeigt die UI nur den generischen Zustand `Not available`.

## LLM Compare

Wenn `LLM Compare` in `Model Ops` eingeschaltet wird:

- bleibt die produktive Summary weiter beim primären Ollama-Modell
- laufen zusätzlich alle in `local_config.json` konfigurierten Compare-Modelle sequenziell
- wird pro aktivierter Session genau eine Exportdatei in `compare_exports/` aufgebaut
- zeigt die UI zusätzlich laufenden Compare-Fortschritt im Header, in der Summary-Queue und in `Model Ops`
- `Model Ops` zeigt zusätzlich einen Compare-Diagnostics-Block mit letztem Status, letzter Dauer und letztem Fehler pro Modell
- der Compare-Bereich in `Model Ops` zeigt außerdem Primär-Summary-Zahlen der aktiven Session, damit `0` bei Compare verständlich bleibt
- läuft der Compare-Teil in einem eigenen Hintergrund-Worker und blockiert den Summary-Worker nicht mehr
- die normale Primary-Summary nutzt einen deutlich höheren eigenen Ollama-Timeout als der Web-Fetch
- der Compare-Teil hat einen eigenen Timeout in `local_config.json`
- Summary und Compare teilen sich intern einen gemeinsamen Ollama-Lock; es läuft also nie mehr als ein lokaler Modell-Call gleichzeitig
- einzelne Compare-Timeouts werden als Fehler protokolliert und lassen die Session weiterlaufen
- fehlgeschlagene Modellläufe zählen als erledigte Compare-Schritte und blockieren die Session nicht dauerhaft
- verglichen werden nur Summaries, die während der aktiven Compare-Session entstanden sind

Die Datei ist bewusst frontier-modell-freundlich formatiert:

- ein Session-Block mit Modellliste
- danach pro Artikel ein XML-artiger Block
- darin URL, Titel, Quelle und die Summary jedes beteiligten Modells

Der Compare-Modus ist absichtlich optional, weil er die Summary-Laufzeit deutlich erhöht.
