# Consultation Dates

This pipeline consists of two main parts:

- A **matcher** which finds mentions of _consultation events_ (more details below)
- A **date parser** (see the corresponding pipeline) that links a date to those events

## Usage

!!! note

    It is designed to work **ONLY on consultation notes** (`CR-CONS`), so please filter accordingly before proceeding.

```python
import spacy


nlp = spacy.blank("fr")
nlp.add_pipe("eds.sentences")
nlp.add_pipe(
    "eds.normalizer",
    config=dict(
        lowercase=True,
        accents=True,
        quotes=True,
        pollution=False,
    ),
)
nlp.add_pipe("eds.consultation_dates")

text = "XXX \n" "Objet : Compte-Rendu de Consultation du 03/10/2018. \n" "XXX "

doc = nlp(text)

doc.spans["consultation_dates"]
# Out: [Consultation du 03/10/2018]

doc.spans["consultation_dates"][0]._.consultation_date
# Out: datetime.datetime(2018, 10, 3, 0, 0)
```

## Consultation events

Three main families of terms are available by default to extract those events.

### The `consultation_mention` terms

This list contains terms directly refering to consultations, such as "_Consultation du..._" or "_Compte rendu du..._".
This list is the only one activated by default since it is fairly precise an not error-prone.

### The `town_mention` terms

This list contains the towns of each AP-HP's hospital. Its goal is to fetch dates mentionned as "_Paris, le 13 décembre 2015_". It has a high recall but poor precision, since those dates can often be dates of letter redaction instea of consultation dates.

### The `document_date_mention` terms

This list contains expressions mentionning the date of creation/edition of a document, such as "_Date du rapport: 13/12/2015_" or "_Signé le 13/12/2015_". As for `town_mention`, it has a high recall but is prone to errors since document date and consultation date aren't necessary similar.

!!! note

    By default, only the `consultation_mention` are used

## Configuration

The pipeline can be configured using the following parameters :

| Parameter               | Explanation                                                | Default                           |
| ----------------------- | ---------------------------------------------------------- | --------------------------------- |
| `consultation_mention`  | Whether to use consultation patterns, or list of patterns  | `True` (use pre-defined patterns) |
| `town_mention`          | Whether to use town patterns, or list of patterns          | `False`                           |
| `document_date_mention` | Whether to use document date patterns, or list of patterns | `False`                           |
| `attr`                  | SpaCy attribute to match on, eg `NORM` or `TEXT`           | `"NORM"`                          |

## Declared extensions

The `eds.consultation_dates` pipeline declares one [SpaCy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes) on the `Span` object :

The `eds.consultation_date` attribute, which is a Python `datetime` object.

## Authors and citation

The `eds.consultation_dates` pipeline was developed by AP-HP's Data Science team.
