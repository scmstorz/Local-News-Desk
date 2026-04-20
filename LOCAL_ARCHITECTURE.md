# Local News Stack

## Ziel

Diese lokale Version ersetzt den bisherigen Google-Stack vollständig:

- kein Firestore
- keine Cloud Function
- kein Vertex AI / Gemini
- lokale Speicherung in `SQLite`
- lokale Summarisierung über `Ollama`
- neues Frontend als eigenständige Datei `local-news-app.html`

Die bestehende `claude-news-app.html` bleibt unverändert als Referenz erhalten.

## Produktfluss

### Feed

- Start immer im Feed
- linearer schneller Flow
- sichtbar: `Titel`, `Quelle`, `Zeit`
- Aktionen:
  - `Weiter` -> negatives Label
  - `Zusammenfassen` -> positives Label
- `Zusammenfassen` läuft im Hintergrund und springt sofort zum nächsten Artikel
- früheres `Weiter` kann durch späteres `Zusammenfassen` überschrieben werden
- ein `Inbox-Reset` archiviert alle aktuell offenen Feed-Einträge, ohne sie zu löschen

### Summaries

- eigener ruhiger Review-Flow
- lineare Abarbeitung
- Aktionen:
  - `Interessant`
  - `Doch nicht interessant`

### LLM Compare

- optional schaltbarer Modus in `Model Ops`
- wenn aktiv, erzeugt jeder neue Summary-Run zusätzlich Zusammenfassungen mehrerer lokal konfigurierbarer Modelle
- die normale App zeigt weiterhin nur die Summary des primären Standardmodells
- zusätzlich entsteht pro aktivierter Compare-Session eine eigene lokale Markdown-Datei
- diese Datei enthält pro Artikel URL, Metadaten und alle Modell-Summaries in XML-artig getrennten Blöcken

### Model Ops

- nüchterne Metrikansicht
- Konfusionsmatrix
- Precision / Recall / F1 / Accuracy
- Anzahl neuer Labels seit letztem Training
- explizite Einschätzung, ob Retraining aktuell sinnvoll ist
- explizite Einschätzung, ob das Feed-Modell eher nur fürs Ranking oder schon fürs härtere Filtering taugt
- manueller Retraining-Trigger
- Compare-Diagnostics pro Modell mit letztem Status, letzter Dauer und letztem Fehler
- Compare-Systembereich zeigt zusätzlich Primär-Summary-Zahlen der aktiven Session

## Architekturentscheidungen

### 1. Frontend als einzelne HTML-Datei

Datei: `local-news-app.html`

Gründe:

- keine Build-Chain
- keine Abhängigkeit von npm oder Framework-Tooling
- leicht lokal zu öffnen und zu versionieren
- passt zum bisherigen Single-File-Charakter der App

Wichtig:

Eine reine HTML-Datei kann RSS, SQLite und Ollama nicht sauber alleine abwickeln. Deshalb gibt es bewusst einen kleinen lokalen Backend-Prozess.

### 2. Lokales Backend in Python

Datei: `local_news_backend.py`

Gründe:

- Python ist für RSS, HTTP, SQLite und ML pragmatisch
- SQLite ist in Python standardmäßig verfügbar
- lokale Datenverarbeitung und Background-Worker sind einfach umsetzbar
- spätere Training-Pipelines lassen sich direkt ergänzen

### 3. SQLite statt Firestore

Datei: `local-news.db`

Gründe:

- lokal, billig, robust
- keine externe Infrastruktur
- gutes Match für das überschaubare Datenmodell
- einfache Auswertungen für Training und Modellmetriken

Zentrale Tabellen:

- `articles`
- `article_events`
- `model_runs`
- `app_state`

### 4. Background-Queue für Summaries

Summaries werden nicht im UI-Thread erzeugt, sondern im Backend asynchron:

- Feed-Aktion markiert Artikel als `queued`
- Worker holt den Artikel
- Text wird extrahiert
- Ollama erzeugt Titel und Summary
- Ergebnis landet als `ready` in der DB

Gründe:

- schneller Feed-Flow
- UI blockiert nicht
- Queue ist sichtbar und nachvollziehbar

### 5. Optionaler Mehrmodell-Vergleich

`LLM Compare` läuft getrennt vom normalen Summary-Worker in einem eigenen Background-Worker.

Verhalten:

- Primärmodell erzeugt die produktive Summary
- zusätzliche Compare-Modelle laufen danach sequenziell in einer separaten Compare-Queue
- Compare-Fehler dürfen die produktive Summary nicht beschädigen
- Ergebnisse werden sowohl in SQLite als auch in einer Exportdatei pro Compare-Session abgelegt
- der laufende Compare-Fortschritt wird separat in `app_state` gehalten und in der UI sichtbar gemacht
- Compare läuft in einem eigenen Hintergrund-Worker und blockiert den Summary-Worker nicht
- einzelne Modell-Timeouts werden als Vergleichsfehler geloggt und die Session läuft weiter
- fehlgeschlagene Modellläufe werden in `llm_compare_results` mit Status gespeichert und zählen als abgeschlossene Schritte
- eine Compare-Session verarbeitet nur Summaries, die seit `enabled_at` dieser Session erzeugt wurden
- `Model Ops` liest Compare-Diagnostics aus den letzten gespeicherten Modellläufen einer Session

Gründe:

- Modellvergleich ohne Eingriff in den normalen Lesefluss
- spätere qualitative Bewertung durch ein Frontier-Modell
- reproduzierbare lokale Benchmark-Sessions

### 6. Zwei Lernstufen statt eines unscharfen Gesamtsignals

#### Stufe 1: Feed Recommendation

Frage:

Soll ein Artikel überhaupt zusammengefasst werden?

Labels:

- `skip`
- `summarize`

Zusätzlicher nicht-trainierender Zustand:

- `archived`

#### Stufe 2: Summary Interest

Frage:

War die erzeugte Summary am Ende relevant?

Labels:

- `interesting`
- `not_interesting`

Gründe:

- trennt Vorfilterung von inhaltlicher Relevanz
- sauberere Trainingsdaten
- später unterschiedliche Modelle oder Schwellen möglich

### 7. Inbox-Reset statt Löschen

Offene Feed-Einträge werden bei Bedarf nicht gelöscht, sondern archiviert.

Gründe:

- der aktive Feed bleibt kontrollierbar, auch wenn mehr Artikel hereinkommen als man labeln kann
- Trainings- und Historiedaten bleiben vollständig erhalten
- Feed-Cutoff ist operativ nützlich, ohne Informationsverlust zu erzeugen

### 8. Model-Ops mit operativer Einschätzung

Die Kennzahlen werden nicht nur roh angezeigt, sondern in eine kurze operative Einschätzung übersetzt.

Beispiele:

- `Mit Retraining noch warten`
- `Neues Training lohnt sich`
- `Nur als Ranking-Signal nutzen`
- `Noch nicht für hartes Filtering geeignet`

Gründe:

- Metriken allein sind im Alltag zu abstrakt
- die Entscheidung "jetzt retrainen oder nicht" soll direkt lesbar sein
- der aktuelle Produktstatus des Modells wird klarer

## Technologieentscheidungen

### Backend / Runtime

- Python 3
- Flask für das lokale HTTP-API
- `sqlite3` für Speicherung
- `requests` für HTTP
- `feedparser` für RSS
- `trafilatura` plus `BeautifulSoup` als Fallback für Textextraktion
- `googlenewsdecoder` für Google-News-Links

### Konfigurationsdatei

Datei: `local_config.json`

Dort liegen die produktnahen Einstellungen:

- RSS-Feed-Liste
- Ollama-Modell
- Ollama-Base-URL
- Compare-Modelle und Exportpfad
- Compare-Timeout pro Modellaufruf
- Polling-Intervalle
- Server- und Speicherpfade

Gründe:

- Feeds und Modellwahl sollen nicht im Code versteckt sein
- spätere Experimente mit anderen Feeds oder Modellen werden einfacher
- lokale Anpassungen sind klarer von Implementierungslogik getrennt

Reihenfolge:

- Standardwerte im Code
- `local_config.json`
- optionale Umgebungsvariablen als Override

### LLM

- Ollama lokal
- Default-Modell: `qwen3.5:35b`

Grund:

- lokal ausführbar
- keine laufenden Cloud-Kosten
- semantisch stärker als kleine lokale Modelle

Hinweis:

Das Modell ist per `OLLAMA_MODEL` überschreibbar.

### Klassifikator

Erster Ansatz:

- `TF-IDF`-Vektorisierung
- `LogisticRegression`

Gründe:

- transparent
- schnell trainierbar
- robust bei kleinen und mittelgroßen Labelmengen
- gute Baseline für Konfusionsmatrix und F1
- Scores sind leicht interpretierbar

Aktuelle Produktentscheidung:

- solange das Feed-Modell noch schwach ist, wird es primär als Ranking-Signal behandelt
- hartes Filtering bleibt optional und experimentell

## Warum zunächst **kein** Embedding-First-Ansatz?

Embeddings sind fachlich interessant und wahrscheinlich später sinnvoll, aber nicht die beste erste Stufe.

### Warum die aktuelle Baseline zuerst sinnvoll ist

- kleine Datensätze profitieren oft stärker von einfachen Modellen
- TF-IDF + Logistic Regression ist leicht debugbar
- Fehlklassifikationen lassen sich einfacher erklären
- Training ist billig und schnell
- Metriken sind direkt vergleichbar

### Wann Embeddings sinnvoll werden

Embeddings lohnen sich besonders, wenn:

- die Labelmenge größer wird
- Titel semantisch ähnlicher werden, aber andere Wörter verwenden
- Quellen und Themen stärker variieren
- später Ranking oder Ähnlichkeitssuche relevant wird

### Sinnvolle spätere Ausbaustufen

1. Embeddings nur als zusätzliches Feature für den Feed-Klassifikator
2. Embeddings für Ähnlichkeitssuche und Cluster
3. Embedding-basiertes Reranking vor dem Klassifikator
4. eigener Retrieval- oder Active-Learning-Loop

Die Grundentscheidung hier ist bewusst:

Erst ein transparentes, kleines Basismodell mit guten Metriken. Danach erst semantische Features.

## Designentscheidungen im Frontend

### 1. Desktop first

Gründe:

- Ollama läuft lokal primär am Desktop
- schneller Arbeitsflow ist wichtiger als mobile Anpassung
- größere Flächen helfen im Model-Ops-Bereich

### 2. Zwei unterschiedliche Nutzungstempi

Feed:

- schnell
- binäre Entscheidungen
- wenig visuelle Ablenkung

Summary Review:

- ruhiger
- lesefreundlicher
- mehr Raum für Text

### 3. Modelltransparenz statt sofortigem Autopilot

Im Feed wird von Anfang an sichtbar:

- Modellentscheidung
- Wahrscheinlichkeit

Zusätzlich gibt es den Toggle:

- `Alle`
- `Nur zeigenswert`

Grund:

Das System soll zuerst Vertrauen aufbauen, bevor es den Feed aggressiv filtert.

### 4. Feed auf schnelle Erfassung getrimmt

Der Feed-Titel wurde bewusst kleiner gehalten als die Summary-Headline.

Gründe:

- Feed ist ein Scan-Interface, kein Lesemodus
- kürzere Blickzeit pro Artikel
- bessere Taktung im linearen Entscheidungsfluss

### 5. Keine Newsletter-Altlasten

Der frühere `NL`-Pfad wurde bewusst entfernt.

Gründe:

- entspricht nicht mehr dem realen Nutzungsszenario
- reduziert Komplexität
- macht Datenmodell und UX klarer

## Datenmodell

### Tabelle `articles`

Enthält den aktuellen Zustand eines Artikels:

- Metadaten aus RSS
- Feed-Entscheidung
- optional archivierter Feed-Status nach Inbox-Reset
- Summary-Status
- Summary-Inhalt
- Summary-Feedback
- optionale Modellvorhersage für den Feed

### Tabelle `article_events`

Enthält die Ereignishistorie:

- Ingestion
- Skip
- Summary requested
- Summary generated
- Summary failed
- Summary feedback
- Feed archiviert

Grund:

Zustand allein reicht nicht für spätere Modellarbeit. Für Training, Audits und Debugging ist der Ereignisverlauf wertvoll.

### Tabelle `model_runs`

Enthält:

- Trainingszeitpunkt
- Zielmodell
- Labelmenge
- Train/Test-Split
- Accuracy / Precision / Recall / F1
- Konfusionsmatrix

Grund:

Model-Ops muss nachvollziehbar und historisierbar bleiben.

### Tabelle `llm_compare_sessions`

Enthält:

- Aktivierungszeitpunkt
- Deaktivierungszeitpunkt
- Exportpfad
- Primärmodell
- Compare-Modellliste

Grund:

Der Mehrmodell-Vergleich soll als klar abgegrenzte Session nachvollziehbar bleiben.

### Tabelle `llm_compare_results`

Enthält:

- Session
- Artikel
- Modellname
- Status
- Dauer des letzten Modelllaufs
- Summary-Titel
- Summary-Text
- optionaler Fehlertext

Grund:

Die Compare-Ergebnisse sollen lokal analysierbar bleiben, auch wenn die Markdown-Datei später extern ausgewertet wird.

### Exportformat von `LLM Compare`

Jede aktivierte Compare-Phase erzeugt genau eine Datei in `compare_exports/`.

Die Struktur ist bewusst simpel und paste-freundlich:

```xml
<compare_session enabled_at="..." primary_model="qwen3.5:35b">
<models>
  <model>qwen3.5:35b</model>
  <model>gemma4:31b</model>
  <model>gpt-oss:20b</model>
  <model>nemotron-3-nano:30b</model>
</models>

<article_compare article_id="...">
  <article_title>...</article_title>
  <source>...</source>
  <url>...</url>
  <model_summary model="qwen3.5:35b" status="ok">
    <summary_title>...</summary_title>
    <summary_text>...</summary_text>
  </model_summary>
  ...
</article_compare>
</compare_session>
```

Grund:

- online Frontier-Modelle sollen die Datei direkt vergleichen können
- die Datei soll auch ohne zusätzliches Tooling lesbar bleiben
- Session- und Artikelgrenzen müssen klar markiert sein

## Legacy-Import

Die alten Feed-Präferenzen aus der früheren Browser-Version wurden als einmaliger Migrationspfad importierbar gemacht.

Wichtig:

- die Altlabels stammen primär aus dem alten `localStorage`-Tracking
- der Import ist Migrationslogik, kein Kernbestandteil des neuen Produktflows
- die UI dafür kann nach erfolgreicher Migration wieder reduziert werden

Grund:

So konnte das erste Feed-Modell sofort mit historischen Präferenzdaten statt nur mit neuen Labels gestartet werden.

## Aktueller Kompromiss

Die erste lokale Version ist bewusst pragmatisch:

- solides lokales Fundament
- einfacher Start
- nachvollziehbare Baseline
- sauberer Weg für spätere Embeddings, bessere Features und aggressivere Filterung

Sie ist nicht als endgültige Endarchitektur gedacht, sondern als robuste, unabhängige und ausbaufähige lokale Basis.
