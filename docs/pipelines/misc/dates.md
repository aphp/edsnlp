# Dates

The `eds.dates` pipeline's role is to detect and normalize dates within a medical document.
We use simple regular expressions to extract date mentions, and apply the [`dateparser` library](https://dateparser.readthedocs.io/en/latest/index.html)
for the normalisation.

!!! warning

    The ``dates`` pipeline is still in active development and has not been rigorously validated.
    If you come across a date expression that goes undetected, please file an issue !

## Scope

The `eds.dates` pipeline finds absolute (eg `23/08/2021`) and relative (eg `hier`, `la semaine dernière`) dates alike.

If the date of edition (via the `doc._.note_datetime` extension) is available, relative (and "year-less") dates will be normalized
using the latter as base. On the other hand, if the base is unknown, the normalisation will follow the pattern :
`TD±<number-of-days>`, positive values meaning that the relative date mentions the future (`dans trois jours`).

Since the extension `doc._.note_datetime` cannot be set before applying the `dates` pipeline, we defer the normalisation step until the `span._.dates` attribute is accessed.

See the [tutorial](/home/tutorials/detecting-dates.md) for a presentation of a full pipeline featuring the `eds.dates` component.

## Usage

```python
import spacy

from datetime import datetime

nlp = spacy.blank("fr")
nlp.add_pipe("eds.dates")

text = (
    "Le patient est admis le 23 août 2021 pour une douleur à l'estomac. "
    "Il lui était arrivé la même chose il y a un an."
)

doc = nlp(text)

dates = doc.spans["dates"]
dates
# Out: [23 août 2021, il y a un an]

dates[0]._.date
# Out: "2021-08-23"

dates[1]._.date
# Out: "TD-365"

doc._.note_datetime = datetime(2021, 8, 27)

dates[1]._.date
# Out: "2010-08-27"
```

## Declared extensions

The `eds.dates` pipeline declares two [SpaCy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes) on the `Span` object :

1. The `date_parsed` attribute is a Python `datetime` object, used internally by the pipeline.
2. The `date` attribute is a property that displays a normalised human-readable string for the date.

## Configuration

The pipeline can be configured using the following parameters :

| Parameter        | Explanation                                      | Default                           |
| ---------------- | ------------------------------------------------ | --------------------------------- |
| `no_year`        | Date patterns without year, eg `le 5 août`       | `None` (use pre-defined patterns) |
| `year_only`      | Date patterns with only the year, eg `en 2018`   | `None` (use pre-defined patterns) |
| `no_day`         | Date patterns without day, eg `en mars 2018`     | `None` (use pre-defined patterns) |
| `absolute`       | Absolute date patterns, eg `le 5 août 2020`      | `None` (use pre-defined patterns) |
| `relative`       | Relative date patterns, eg `hier`)               | `None` (use pre-defined patterns) |
| `full`           | Full date patterns, eg `2020-10-23`              | `None` (use pre-defined patterns) |
| `current`        | "Current" date patterns, eg `ce jour`            | `None` (use pre-defined patterns) |
| `false_positive` | Some false positive patterns to exclude          | `None` (use pre-defined patterns) |
| `on_ents_only`   | Whether to look for dates around entities only   | `False`                           |
| `attr`           | SpaCy attribute to match on, eg `NORM` or `TEXT` | `"NORM"`                          |

## Authors and citation

The `eds.dates` pipeline was developed by AP-HP's Data Science team.
