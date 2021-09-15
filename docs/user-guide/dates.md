# Dates

The `dates` pipeline's role is to detect and normalize dates within a medical document.
We use simple regular expressions to extract date mentions, and apply the `[dateparser` library](https://dateparser.readthedocs.io/en/latest/index.html)
for the normalization.

## Scope

The `dates` pipeline finds absolute (eg `23/08/2021`) and relative (eg `hier`, `la semaine dernière`) dates alike.

If the date of edition (via the `doc._.note_datetime` extension) is available, relative dates will be normalized
using the latter as base. On the other hand, if the base is unknown, the normalization will follow the pattern :
`TD±<number-of-days>`, positive values meaning that the relative date mentions the future (`dans trois jours`).

## Usage

```python
import spacy
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("dates")  # exposed via edsnlp.components

text = (
    "Le patient est admis le 23 août 2021 pour une douleur à l'estomac. "
    "Il lui était arrivé la même chose il y a un an."
)

doc = nlp(text)

dates = doc.spans["dates"]
dates
# Out: [23 août 2021, il y a un an]

dates[0].label_
# Out: "2021-08-23"

dates[1].label_
# Out: "TD-365"
```

## Authors and citation

The `dates` pipeline was developed at the Data and Innovation unit, IT department, AP-HP.
