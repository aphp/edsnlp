# Dates

The `eds.dates` pipeline's role is to detect and normalise dates within a medical document.
We use simple regular expressions to extract date mentions.

## Scope

The `eds.dates` pipeline finds absolute (eg `23/08/2021`) and relative (eg `hier`, `la semaine dernière`) dates alike.

See the [tutorial](../../tutorials/detecting-dates.md) for a presentation of a full pipeline featuring the `eds.dates` component.

## Usage

```python
import spacy

import pendulum

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

dates[0]._.date.parse()
# Out: DateTime(2021, 8, 23, 0, 0, 0, tzinfo=Timezone('Europe/Paris'))

dates[1]._.date.parse()
# Out: Duration(years=-1)

note_datetime = pendulum.datetime(2021, 8, 27, tz="Europe/Paris")

dates[1]._.date.parse(note_datetime=note_datetime)
# Out: DateTime(2020, 8, 27, 0, 0, 0, tzinfo=Timezone('Europe/Paris'))
```

## Declared extensions

The `eds.dates` pipeline declares one [spaCy extension](https://spacy.io/usage/processing-pipelines#custom-components-attributes) on the `Span` object: the `date` attribute contains a parsed version of the date.

## Configuration

The pipeline can be configured using the following parameters :

| Parameter        | Explanation                                      | Default                           |
| ---------------- | ------------------------------------------------ | --------------------------------- |
| `absolute`       | Absolute date patterns, eg `le 5 août 2020`      | `None` (use pre-defined patterns) |
| `relative`       | Relative date patterns, eg `hier`)               | `None` (use pre-defined patterns) |
| `false_positive` | Some false positive patterns to exclude          | `None` (use pre-defined patterns) |
| `on_ents_only`   | Whether to look for dates around entities only   | `False`                           |
| `attr`           | spaCy attribute to match on, eg `NORM` or `TEXT` | `"NORM"`                          |

## Authors and citation

The `eds.dates` pipeline was developed by AP-HP's Data Science team.
