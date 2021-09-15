# QuickUMLS

[Georgetown's implementations](https://github.com/Georgetown-IR-Lab/QuickUMLS) of QuickUMLS is not fully Spacy 3.0 compatible. We therefore developed a `QuickUMLS` component.

## Installation

You need to acquire a [UMLS license](https://uts.nlm.nih.gov/uts/signup-login) and [install the metathesaurus](https://www.nlm.nih.gov/research/umls/index.html) to be able to use this component.

## Usage

```python
import spacy
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe(
    "quickumls", config=dict(distribution="path/to/distribution")
)  # exposed via edsnlp.components

text = "Le patient est admis pour des douleurs dans le bras droit."

doc = nlp(text)

doc.ents
# Out: (douleurs,)
```

## Authors and citation

The `quickumls` pipeline was developed at the Data and Innovation unit, IT department, AP-HP, **relying heavily on previous work by Georgetown**.
