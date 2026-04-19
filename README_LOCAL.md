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
- optionale `LLM Compare`-Modelle
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

## LLM Compare

Wenn `LLM Compare` in `Model Ops` eingeschaltet wird:

- bleibt die produktive Summary weiter beim primären Ollama-Modell
- laufen zusätzlich alle in `local_config.json` konfigurierten Compare-Modelle sequenziell
- wird pro aktivierter Session genau eine Exportdatei in `compare_exports/` aufgebaut

Die Datei ist bewusst frontier-modell-freundlich formatiert:

- ein Session-Block mit Modellliste
- danach pro Artikel ein XML-artiger Block
- darin URL, Titel, Quelle und die Summary jedes beteiligten Modells

Der Compare-Modus ist absichtlich optional, weil er die Summary-Laufzeit deutlich erhöht.
