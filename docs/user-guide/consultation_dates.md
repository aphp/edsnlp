# Consultation Dates

This pipeline consists of two main parts:
- A **matcher** which finds mentions of *consultation events* (more details below)
- A **date parser** (see the corresponding pipeline) that links a date to those events

```{note}
It is designed to work **ONLY on consultation notes** (`CR-CONS`), so please filter accordingly before proceeding.
```

## Consultation events

Three main families of terms are available by default to extract those events.

### The `consultation_mention` terms

This list contains terms directly refering to consultations, such as "*Consultation du...*" or "*Compte rendu du...*".
This list is the only one activated by default since it is fairly precise an not error-prone.

### The `town_mention` terms

This list contains the towns of each AP-HP's hospital. Its goal is to fetch dates mentionned as "*Paris, le 13 décembre 2015*". It has a high recall but poor precision, since those dates can often be dates of letter redaction instea of consultation dates.

### The `document_date_mention` terms

This list contains expressions mentionning the date of creation/edition of a document, such as "*Date du rapport: 13/12/2015*" or "*Signé le 13/12/2015*". As for `town_mention`, it has a high recall but is prone to errors since document date and consultation date aren't necessary similar.

```{note}
By default, only the `consultation_mention` are used
```

## Declared extensions

The `consultation_dates` pipeline declares one [Spacy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes) on the `Span` object :

The `consultation_date` attribute, which is a Python `datetime` object

## Usage

As mentionned above, please make sure to only use this pipeline with consultation notes.

```python
import spacy
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("sentences")
nlp.add_pipe(
    "normalizer",
    config=dict(
        lowercase=True,
        accents=True,
        quotes=True,
        pollution=False,
    ),
)
nlp.add_pipe("consultation_dates")

text = "XXX " "Objet : Compte-Rendu de Consultation du 03/10/2018. " "XXX "

doc = nlp(text)

doc.spans["consultation_dates"]
# Out: [Consultation du 03/10/2018]

doc.spans["consultation_dates"]._.consultation_date
# Out: datetime.datetime(2018, 10, 3, 0, 0)
```
