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

`qwen3.6:latest`

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
OLLAMA_MODEL=qwen3.6:latest
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
- `Not available`-Fälle erscheinen in der Summary-Liste, damit der Originalartikel bei Bedarf noch geöffnet werden kann.
- hängengebliebene `processing`-Summaries werden nach einem Recovery-Timeout automatisch auf `failed` gesetzt.
- Der Feed blendet sehr ähnliche Meldungen aus anderen Quellen standardmäßig zusammen und zeigt nur einen Hauptartikel pro Story.
- Die Ähnlichkeitsgruppierung läuft als Hintergrund-Snapshot; der Feed-Request liest diesen Snapshot nur noch aus und bleibt dadurch beim Reload deutlich schneller.
- Der Toggle `Empfohlen` filtert inzwischen client-seitig auf Basis bereits geladener Feed-Daten; das Umschalten erzeugt keinen zusätzlichen Backend-Request mehr.
- Der Feed hält einen lokalen Verlauf, damit `Zurück` und Pfeil links auch nach `Weiter` oder `Zusammenfassen` weiter funktionieren.
- `Model Ops` zeigt während eines Trainingslaufs sofort einen sichtbaren Laufstatus statt nur still auf neue Zahlen zu springen.

## LLM Compare

Wenn `LLM Compare` in `Model Ops` eingeschaltet wird:

- bleibt die produktive Summary weiter beim primären Ollama-Modell
- laufen zusätzlich alle in `local_config.json` konfigurierten Compare-Modelle sequenziell
- wird pro aktivierter Session genau eine Exportdatei in `compare_exports/` aufgebaut
- zeigt die UI zusätzlich laufenden Compare-Fortschritt im Header, in der Summary-Queue und in `Model Ops`
- `Model Ops` zeigt zusätzlich einen Compare-Diagnostics-Block mit letztem Status, letzter Dauer und letztem Fehler pro Modell
- `Model Ops` zeigt bei trainierten Targets zusätzlich einen direkten Vorher/Nachher-Vergleich der letzten Metriken (`Accuracy`, `Precision`, `Recall`, `F1`)
- `Model Ops` ergänzt diesen Vergleich inzwischen um eine kurze verbale Einschätzung wie `Eher besser`, `Gemischt` oder `Eher seitwaerts`
- ein aufklappbarer Erklärbereich beschreibt für Laien, was `Precision`, `Recall` und `F1` in diesem Produktkontext bedeuten
- der Compare-Bereich in `Model Ops` trennt Primär-Summary-Zahlen der aktiven Session sauber von Compare-Modellläufen
- Primär-Fehler werden dort grob in `Quelle/Text` versus `Ollama` getrennt, Compare-Fehler separat als Modelllauf-Fehler gezählt
- der sichtbare Hauptstatus in `Model Ops` ist bewusst in Klartext formuliert; technische Zähler liegen nur noch unter `Technische Details`
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

## Ähnliche Feed-Meldungen

Der Feed gruppiert sehr ähnliche Headlines bewusst eher streng, aber inzwischen etwas toleranter gegen leichte Titelvarianten, damit dieselbe Story nicht mehrfach aus verschiedenen Quellen auftaucht.

Verhalten:

- pro Story bleibt nur ein sichtbarer Hauptartikel im Feed
- der Feed zeigt bei Bedarf `+N ähnliche Meldungen`
- oben zeigt die UI zusätzlich, wie viele ähnliche Meldungen aktuell in wie vielen Gruppen unterdrückt werden
- über `Inbox kompaktieren` lässt sich die aktuelle Inbox optional einmalig konservativ aufräumen
- wenn du den Hauptartikel `Weiter` klickst oder `Zusammenfassen` wählst, werden die sehr ähnlichen Pending-Dubletten ebenfalls aus dem aktiven Feed entfernt
- nach einem bereits getroffenen Entscheid werden später eingehende, sehr ähnliche Nachzügler automatisch aus dem Feed genommen

Die Logik ist bewusst konservativ:

- nur sehr ähnliche Headlines werden zusammengezogen, aber leichte Umformulierungen derselben Story werden inzwischen etwas eher erkannt
- verwandte Follow-ups mit neuem Winkel sollen weiterhin durchkommen

Performance:

- die Similarity-Bildung über viele Pending-Artikel läuft nicht mehr direkt im Feed-Request
- stattdessen wird ein Snapshot im Hintergrund vorbereitet und nur im laufenden Backend-Prozess im Speicher gehalten
- der Feed fällt nur dann kurzfristig auf einen ungruppierten Zustand zurück, wenn noch kein aktueller Snapshot vorhanden ist
