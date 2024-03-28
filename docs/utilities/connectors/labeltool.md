# LabelTool Connector

LabelTool is an in-house module enabling rapid annotation of pre-extracted entities.

We provide a ready-to-use function that converts a list of annotated spaCy documents into a `pandas` DataFrame that is readable to LabelTool.

```python
import edsnlp, edsnlp.pipes as eds

from edsnlp.connectors.labeltool import docs2labeltool

corpus = [
    "Ceci est un document médical.",
    "Le patient n'est pas malade.",
]

# Instantiate the spacy pipeline
nlp = edsnlp.blank("fr")
nlp.add_pipe(eds.sentences())
nlp.add_pipe(eds.matcher(terms=dict(medical="médical", malade="malade")))
nlp.add_pipe(eds.negation())

# Convert all BRAT files to a list of documents
docs = nlp.pipe(corpus)

df = docs2labeltool(docs, extensions=["negation"])
```

The results:

| note_id | note_text                     | start | end | label   | lexical_variant | negation |
| ------- | ----------------------------- | ----- | --- | ------- | --------------- | -------- |
| 0       | Ceci est un document médical. | 21    | 28  | medical | médical         | False    |
| 1       | Le patient n'est pas malade.  | 21    | 27  | malade  | malade          | True     |
