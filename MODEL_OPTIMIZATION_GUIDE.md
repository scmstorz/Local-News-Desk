# Modell-Optimierung fuer den Feed-Praediktor

## Worum es hier geht

Diese App versucht vorherzusagen, welche Feed-Artikel fuer dich wahrscheinlich
relevant genug sind, um eine lokale Summary erzeugen zu lassen.

Das Modell entscheidet **nicht** ueber Wahrheit oder Qualitaet eines Artikels.
Es versucht nur zu lernen:

- Welche Artikel du typischerweise mit `Zusammenfassen` markierst
- Welche Artikel du typischerweise mit `Weiter` ueberspringst

Das Ziel ist deshalb **nicht Perfektion**, sondern ein brauchbares
Empfehlungssignal:

- moeglichst wenige wirklich interessante Artikel uebersehen
- aber auch nicht so viele irrelevante Artikel als `Empfohlen` markieren, dass
  du wieder zu viel lesen musst

## Das Grundproblem

Im Feed sind die Klassen ungleich verteilt:

- viel mehr Artikel sind am Ende `Weiter`
- deutlich weniger Artikel sind `Zusammenfassen`

Deshalb ist eine hohe `Accuracy` allein fast wertlos.

Ein dummes Modell koennte fast alles auf `nicht interessant` setzen und haette
trotzdem oft eine scheinbar gute Trefferquote. Das hilft im Alltag aber nicht.

## Die vier Faelle der Konfusionsmatrix

Die Konfusionsmatrix ist die wichtigste Basis fuer die Bewertung.

Sie besteht aus vier Faellen:

### 1. True Positive

Das Modell sagt: `Empfohlen`
Und du entscheidest spaeter ebenfalls: `Zusammenfassen`

Das ist ein echter Treffer.

### 2. False Positive

Das Modell sagt: `Empfohlen`
Du entscheidest spaeter aber: `Weiter`

Das ist ein Fehlalarm.

Praktische Bedeutung:
Du liest oder beachtest einen Artikel, der sich fuer dich am Ende nicht lohnt.

### 3. True Negative

Das Modell sagt: `nicht empfohlen`
Und du entscheidest spaeter ebenfalls: `Weiter`

Das ist korrekt aussortiert.

### 4. False Negative

Das Modell sagt: `nicht empfohlen`
Du entscheidest spaeter aber doch: `Zusammenfassen`

Das ist der wichtigste verpasste Treffer.

Praktische Bedeutung:
Die App haette dir einen guten Artikel beinahe weggenommen.

## Welche Fehler schlimmer sind

In diesem Projekt gilt:

- `False Negatives` sind schlimmer als `False Positives`

Warum?

Wenn ein guter Artikel uebersehen wird, ist das fuer dich teurer als ein
zusaetzlicher Fehlalarm.

Trotzdem sind `False Positives` nicht egal:

- zu viele Fehlalarme machen `Empfohlen` unbrauchbar
- dann musst du wieder zu viel lesen

Die Optimierung sucht deshalb **keine extreme Recall-Loesung**, sondern eine
brauchbare Balance mit leichter Bevorzugung gegen uebersehene Treffer.

## Die wichtigsten Metriken

### Precision

Frage:

Wenn das Modell `Empfohlen` sagt, wie oft liegt es damit richtig?

Formel:

`Precision = True Positives / (True Positives + False Positives)`

Praktische Bedeutung:

- hohe Precision = wenig Fehlalarme
- niedrige Precision = `Empfohlen` ist verrauscht

### Recall

Frage:

Von allen wirklich interessanten Artikeln, wie viele findet das Modell?

Formel:

`Recall = True Positives / (True Positives + False Negatives)`

Praktische Bedeutung:

- hoher Recall = wenige gute Artikel gehen verloren
- niedriger Recall = das Modell verpasst viele gute Kandidaten

### F1

F1 ist ein kombinierter Wert aus Precision und Recall.

Praktische Bedeutung:

- F1 steigt, wenn Precision und Recall gemeinsam besser werden
- F1 sinkt, wenn eine Seite stark leidet

F1 ist fuer diese App die wichtigste **Kurzmetrik fuer die Gesamtbalance**.

### Accuracy

Accuracy misst den Anteil aller richtigen Vorhersagen.

Formel:

`Accuracy = (True Positives + True Negatives) / alle Faelle`

Warum sie hier oft irrefuehrend ist:

- es gibt viel mehr negative als positive Faelle
- deshalb kann Accuracy gut aussehen, obwohl das Modell kaum gute Artikel findet

Accuracy ist hier also nur ein Nebenwert.

## Was `Precision@10`, `Precision@20`, `Precision@50` bedeuten

Diese Werte beantworten eine alltagsnaehere Frage:

- Wie gut sind die **obersten Empfehlungen**?

Beispiel:

- `Precision@10 = 0.40`

bedeutet:

- Unter den obersten 10 Empfehlungen waren 4 wirklich gute Treffer

Warum das wichtig ist:

Im Alltag liest du nicht die gesamte Modellwelt aus, sondern vor allem die
obersten Kandidaten. Deshalb ist `Precision@K` oft naeher an der realen
Nutzererfahrung als nur globale Metriken.

## Was ein Threshold ist

Das Modell gibt keine direkte Ja/Nein-Antwort aus, sondern eine
Wahrscheinlichkeit, zum Beispiel:

- `0.18`
- `0.44`
- `0.73`

Erst der **Threshold** macht daraus eine Entscheidung:

- ueber dem Threshold -> `Empfohlen`
- zwischen Hauptschwelle und zweiter Schwelle -> `Vielleicht`
- darunter -> `nicht empfohlen`

`Empfohlen` und `Vielleicht` sind in der UI bewusst getrennte Listen.
Wenn du zuerst `Empfohlen` und danach `Vielleicht` durcharbeitest, soll
derselbe Artikel nicht nochmal in der zweiten Liste auftauchen.

Beispiel:

- Hauptschwelle `0.50`: nur deutlich positive Artikel werden empfohlen
- zweite Schwelle `0.35`: grenzwertige Artikel werden als `Vielleicht` markiert

Konsequenz:

- niedrigerer Threshold -> mehr Recall, aber oft weniger Precision
- hoeherer Threshold -> mehr Precision, aber oft weniger Recall

## Wie in dieser App optimiert wird

Der Feed-Praediktor verwendet derzeit:

- `TF-IDF`
- `Logistic Regression`

Das ist bewusst ein einfaches, transparentes Basismodell.

Beim Retraining passieren mehrere Schritte:

1. Alle vorhandenen Feed-Labels werden gesammelt
2. Die Daten werden in Train- und Testdaten aufgeteilt
3. Das Modell wird auf den Trainingsdaten trainiert
4. Auf den Testdaten werden Wahrscheinlichkeiten berechnet
5. Die Entscheidungsschwelle wird automatisch gesucht
6. Die resultierenden Metriken werden gespeichert und in `Model Ops` gezeigt

## Warum die Schwelle nicht einfach immer 0.5 ist

Eine starre Schwelle von `0.5` ist selten optimal.

In dieser App wird die Schwelle beim Retraining deshalb automatisch angepasst.

Die aktuelle Strategie ist:

- eher precision-lastig als recall-lastig
- mit einer zweiten, weicheren `Vielleicht`-Stufe fuer Grenzfaelle

Konkret:

- es wird nicht nur nach maximalem Recall gesucht
- es wird auch nicht nur blind nach F1 gesucht
- stattdessen wird zuerst eine moeglichst treffsichere Hauptschwelle gesucht
- darunter wird eine zweite Schwelle fuer `Vielleicht` gesetzt
- diese zweite Stufe ist kein `Empfohlen + Vielleicht`, sondern nur der
  unsichere Grenzbereich

Das ist noetig, weil fuer diesen Use Case gilt:

- gute Artikel nicht verpassen
- aber `Empfohlen` trotzdem alltagstauglich und moeglichst selektiv halten

## Woran man Fortschritt erkennt

Ein Modell ist **nicht automatisch besser**, nur weil eine Zahl steigt.

Wichtige Signale fuer echten Fortschritt sind:

### Gute Zeichen

- `F1` steigt klar
- `Precision` steigt, ohne dass `Recall` abstuerzt
- oder `Recall` steigt, ohne dass `Precision` kollabiert
- `Precision@10` und `Precision@20` werden besser
- die obersten Empfehlungen wirken im Alltag plausibler

### Warnzeichen

- `Recall` steigt stark, aber `Precision` faellt drastisch
- `Accuracy` steigt, waehrend `Recall` faellt
- `Empfohlen` fuehlt sich subjektiv immer noisiger an
- das Modell markiert ploetzlich fast alles als interessant

## Ein reales Beispiel fuer "scheinbar besser, praktisch schlechter"

Angenommen:

- `Recall` springt von `0.38` auf `0.65`
- `Precision` faellt von `0.23` auf `0.14`

Dann findet das Modell zwar mehr gute Artikel, aber produziert auch sehr viele
zusaetzliche Fehlalarme.

Das kann praktisch schlechter sein, obwohl eine einzelne Metrik steigt.

Deshalb schaut diese App immer auf:

- Konfusionsmatrix
- Precision
- Recall
- F1
- zusaetzlich `Precision@K`

## Was aktuell ein gutes Ergebnis waere

Es gibt keinen universellen magischen Zielwert. Sinnvoll ist ein Fortschritt
dann, wenn:

- `F1` ueber mehrere Trainingslaeufe stabil steigt
- `Precision` nicht zu weit einbricht
- `Recall` trotzdem sichtbar brauchbar bleibt
- `Precision@10` bei den obersten Empfehlungen subjektiv und messbar besser ist

Das beste Modell fuer diese App ist also nicht das mathematisch "perfekteste",
sondern das, das dir den Feed **spuerbar besser vorsortiert**.

## Warum subjektiver Eindruck trotzdem wichtig bleibt

Statistik zeigt, ob sich das Modell verbessert.
Aber sie ersetzt nicht die praktische Frage:

- Sind die Artikel unter `Empfohlen` jetzt tatsaechlich lesenswerter?

Deshalb zaehlen immer zwei Ebenen:

### 1. Metriken

- objektiver Vergleich zwischen Trainingslaeufen

### 2. Alltagseindruck

- fuehlt sich `Empfohlen` nuetzlicher an?
- tauchen dort weniger nervige Fehlalarme auf?
- werden gute Artikel seltener uebersehen?

Erst beides zusammen ergibt ein gutes Urteil.

## Was spaeter noch besser werden kann

Die aktuelle Baseline ist bewusst einfach.

Spaetere sinnvolle Ausbaustufen sind:

- bessere Features
- Embeddings im Hintergrund
- robustere Embedding-Eingaben fuer unterschiedliche Sprachen und Schriftsysteme
- differenziertere Schwellenstrategie
- Auswertung harter Fehlfaelle wie:
  - `nicht empfohlen, aber trotzdem zusammenfassen lassen`
  - `empfohlen, aber sofort uebersprungen`

## Kurzfassung

Wenn man sich nur drei Dinge merken will, dann diese:

1. `False Negatives` sind hier schlimmer als `False Positives`, aber beide sind wichtig.
2. `F1` ist die beste Kurzmetrik fuer die Balance, `Accuracy` ist nur Nebeninformation.
3. Ein Fortschritt ist erst dann wirklich gut, wenn sowohl die Metriken als auch dein Alltagseindruck besser werden.
