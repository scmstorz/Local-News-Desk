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

Danach die App im Browser öffnen:

`http://127.0.0.1:8765/`

Alternativ kann `local-news-app.html` weiterhin direkt geöffnet werden.

Das Frontend spricht standardmäßig mit:

`http://127.0.0.1:8765`

Ein separater Cronjob ist nicht nötig.

Der Backend-Prozess läuft dauerhaft und zieht standardmäßig alle 5 Minuten neue RSS-Einträge nach.

## Tests

Die kleine Regressionssuite laeuft ohne zusaetzliche Testdependency mit `unittest`:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Sie nutzt temporaere SQLite-Datenbanken und deckt aktuell Feed-Aktionen, Feed-Modi, Embedding-Parsing, Summary-Priorisierung, den `S`-Shortcut im Frontend und den HTTP-Auslieferungsvertrag der UI ab.

Der fruehere selbstgebaute Chrome/CDP-Browser-Test wurde durch `tests/test_ui_http_contract.py` ersetzt. Die App wird nun vom Backend ueber `/` und `/app` ausgeliefert; direkte Datei-Nutzung bleibt als Fallback erhalten.

Pflicht fuer weitere Aenderungen:

- Nach jeder Codeaenderung an Backend, Frontend oder Datenmodell diese Suite ausfuehren.
- Wenn ein Bug gefixt wird, nach Moeglichkeit einen passenden Regressionstest in `tests/test_regressions.py` ergaenzen.
- Ein Stand gilt erst als fertig, wenn diese Suite und die einfachen Syntaxchecks gruen sind.

Empfohlener Abschlusscheck:

```bash
.venv/bin/python -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile app.py local_news_backend.py tests/test_regressions.py tests/test_ui_http_contract.py
perl -0ne 'while (m{<script>(.*?)</script>}sg) { print $1 }' local-news-app.html | node --check
git diff --check
```

## Wichtige Dateien

- `local-news-app.html`
- `local_news_backend.py`
- `tests/test_regressions.py`
- `tests/test_ui_http_contract.py`
- `local_config.json`
- `LOCAL_ARCHITECTURE.md`
- `MODEL_OPTIMIZATION_GUIDE.md`
- `requirements.txt`

## Konfiguration

Die lokale Konfiguration liegt in:

`local_config.json`

Dort werden unter anderem gepflegt:

- Google-News-RSS-Feeds
- Ollama-Base-URL
- Ollama-Modell
- Ollama-Embedding-Modell fuer die Aehnlichkeitslogik
- eigener Ollama-Summary-Timeout
- eigener Ollama-Embedding-Timeout
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

Fuer die embedding-basierte Aehnlichkeit solltest du zusaetzlich einmal lokal verfuegbar haben:

```bash
ollama pull nomic-embed-text-v2-moe:latest
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
- `Zusammenfassen` weckt den Summary-Worker sofort; neue Summary-Jobs muessen nicht erst auf das naechste Polling warten.
- Der Feed blendet sehr ähnliche Meldungen aus anderen Quellen standardmäßig zusammen und zeigt nur einen Hauptartikel pro Story.
- Das Feed-Training arbeitet cluster-bewusst: sehr ähnliche Story-Varianten werden vor dem Training zu einem Fall zusammengezogen; `Zusammenfassen` dominiert dabei mehrere `Weiter`-Varianten derselben Story.
- Die Similarity-Logik nutzt zusaetzlich lokale Embeddings, aber nur im Hintergrund. Feed-Reload und Feed-Klicks fuehren selbst keine Embedding-Generierung aus.
- Embeddings behalten Unicode-Titel als Fallback bei; nicht-lateinische Headlines werden dadurch nicht mehr zu leerem Embedding-Input normalisiert.
- Der Embedding-Worker pausiert, solange Summary-Jobs `queued` oder `processing` sind, damit lokale Ollama-Kapazitaet zuerst fuer produktive Summaries genutzt wird.
- Die Ähnlichkeitsgruppierung läuft als Hintergrund-Snapshot; der Feed-Request liest diesen Snapshot nur noch aus und bleibt dadurch beim Reload deutlich schneller.
- Der Toggle `Empfohlen` filtert inzwischen client-seitig auf Basis bereits geladener Feed-Daten; das Umschalten erzeugt keinen zusätzlichen Backend-Request mehr.
- `Empfohlen` und `Vielleicht` sind im Feed getrennte Modi ohne Ueberschneidung; `Vielleicht` zeigt nur Grenzfaelle unterhalb der strengen Empfehlungsschwelle.
- Der Feed hält einen lokalen Verlauf, damit `Zurück` und Pfeil links auch nach `Weiter` oder `Zusammenfassen` weiter funktionieren.
- `Model Ops` zeigt während eines Trainingslaufs sofort einen sichtbaren Laufstatus statt nur still auf neue Zahlen zu springen.
- beim Feed-Retraining wird die Entscheidungsschwelle nicht starr auf `0.5` gelassen, sondern automatisch precision-first optimiert
- das Feed-Modell hat jetzt zwei Stufen:
  - `empfohlen` fuer die strengeren, treffsichereren Treffer
  - `vielleicht` fuer grenzwertige Kandidaten unterhalb der Hauptschwelle
- `Model Ops` zeigt fuer das Feed-Modell deshalb zusaetzlich die aktive `Threshold`-Schwelle, die `Vielleicht`-Schwelle sowie `Precision@10`, `Precision@20` und `Precision@50`
- Feed-Entscheidungen loggen im Event-Store jetzt zusaetzlich einen Snapshot der damals aktiven Modellvorhersage mit
- ein neuer Feed-Lauf wird nicht automatisch live geschaltet: schlechtere Kandidaten werden verworfen, das aktive Modell bleibt dann unveraendert

## LLM Compare

Wenn `LLM Compare` in `Model Ops` eingeschaltet wird:

- bleibt die produktive Summary weiter beim primären Ollama-Modell
- laufen zusätzlich alle in `local_config.json` konfigurierten Compare-Modelle sequenziell
- wird pro aktivierter Session genau eine Exportdatei in `compare_exports/` aufgebaut
- zeigt die UI zusätzlich laufenden Compare-Fortschritt im Header, in der Summary-Queue und in `Model Ops`
- `Model Ops` zeigt zusätzlich einen Compare-Diagnostics-Block mit letztem Status, letzter Dauer und letztem Fehler pro Modell
- `Model Ops` zeigt bei trainierten Targets zusätzlich einen direkten Vorher/Nachher-Vergleich der letzten Metriken (`Accuracy`, `Precision`, `Recall`, `F1`)
- `Model Ops` ergänzt diesen Vergleich inzwischen um eine kurze verbale Einschätzung wie `Eher besser`, `Gemischt` oder `Eher seitwaerts`
- ein aufklappbarer Erklärbereich beschreibt für Laien, was `Precision`, `Recall`, `F1`, `Threshold` und `Precision@K` in diesem Produktkontext bedeuten
- `Model Ops` zeigt zusätzlich die Dauer des letzten Trainingslaufs
- der Compare-Bereich in `Model Ops` trennt Primär-Summary-Zahlen der aktiven Session sauber von Compare-Modellläufen
- Primär-Fehler werden dort grob in `Quelle/Text` versus `Ollama` getrennt, Compare-Fehler separat als Modelllauf-Fehler gezählt
- der sichtbare Hauptstatus in `Model Ops` ist bewusst in Klartext formuliert; technische Zähler liegen nur noch unter `Technische Details`
- läuft der Compare-Teil in einem eigenen Hintergrund-Worker und blockiert den Summary-Worker nicht mehr
- die normale Primary-Summary nutzt einen deutlich höheren eigenen Ollama-Timeout als der Web-Fetch
- der Compare-Teil hat einen eigenen Timeout in `local_config.json`
- Summary und Compare teilen sich intern einen gemeinsamen Ollama-Lock; es läuft also nie mehr als ein lokaler Modell-Call gleichzeitig
- Summary-Jobs haben operativ Vorrang vor Embedding-Jobs; Compare bleibt optional und blockiert den produktiven Summary-Worker nicht
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
- die Aehnlichkeitsgruppierung vergleicht inzwischen nicht nur gegen einen kanonischen Titel, sondern gegen bereits erkannte Cluster-Mitglieder und erwischt dadurch leichte Varianten derselben Story robuster
- wenn verfuegbar, fliesst zusaetzlich eine semantische Aehnlichkeit ueber lokale Ollama-Embeddings ein

Die Logik ist bewusst konservativ:

- nur sehr ähnliche Headlines werden zusammengezogen, aber leichte Umformulierungen derselben Story werden inzwischen etwas eher erkannt
- verwandte Follow-ups mit neuem Winkel sollen weiterhin durchkommen

Performance:

- die Similarity-Bildung über viele Pending-Artikel läuft nicht mehr direkt im Feed-Request
- stattdessen wird ein Snapshot im Hintergrund vorbereitet und nur im laufenden Backend-Prozess im Speicher gehalten
- der Feed fällt nur dann kurzfristig auf einen ungruppierten Zustand zurück, wenn noch kein aktueller Snapshot vorhanden ist
- ein separater Embedding-Worker erzeugt Titel-Embeddings im Hintergrund; auch diese Arbeit laeuft nie im Hot Path des Feed-Klickens
