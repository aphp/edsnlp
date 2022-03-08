# OMOP Connector

We provide a connector between OMOP-formatted dataframes and spaCy documents.

## OMOP-style dataframes

Consider a corpus of just one document:

```
Le patient est admis pour une pneumopathie au coronavirus.
On lui prescrit du paracétamol.
```

And its OMOP-style representation, separated in two tables `note` and `note_nlp` (here with selected columns) :

`note`:

| note_id | note_text                                     | note_datetime |
| ------: | :-------------------------------------------- | :------------ |
|       0 | Le patient est admis pour une pneumopathie... | 2021-10-23    |

`note_nlp`:

| note_nlp_id | note_id | start_char | end_char | note_nlp_source_value | lexical_variant |
| ----------: | ------: | ---------: | -------: | :-------------------- | :-------------- |
|           0 |       0 |         46 |       57 | disease               | coronavirus     |
|           1 |       0 |         77 |       88 | drug                  | paracétamol     |

## Using the connector

The following snippet expects the tables `note` and `note_nlp` to be already defined (eg through PySpark's `toPandas()` method).

```python
import spacy
from edsnlp.connectors.omop import OmopConnector

# Instantiate a spacy pipeline
nlp = spacy.blank("fr")

# Instantiate the connector
connector = OmopConnector(nlp)

# Convert OMOP tables (note and note_nlp) to a list of documents
docs = connector.omop2docs(note, note_nlp)
doc = docs[0]

doc.ents
# Out: [coronavirus, paracétamol]

doc.ents[0].label_
# Out: disease

doc.text == note.loc[0].note_text
# Out: True
```

The object `docs` now contains a list of documents that reflects the information contained in the OMOP-formatted dataframes.
