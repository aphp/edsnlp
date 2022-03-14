# Matcher

EDS-NLP simplifies the matching process by exposing a `eds.matcher` pipeline
that can match on terms or regular expressions.

## Usage

Let us redefine the pipeline :

```python
import spacy

nlp = spacy.blank("fr")

terms = dict(
    covid=["coronavirus", "covid19"],  # (1)
    patient="patient",  # (2)
)

regex = dict(
    covid=r"coronavirus|covid[-\s]?19|sars[-\s]cov[-\s]2",  # (3)
)

nlp.add_pipe("eds.matcher", config=dict(terms=terms, regex=regex, attr="LOWER"))
```

1. Every key in the `terms` dictionary is mapped to a concept.
2. The `eds.matcher` pipeline expects a list of expressions, or a single expression.
3. We can also define regular expression patterns.

This snippet is complete, and should run as is.

## Configuration

The pipeline can be configured using the following parameters :

| Parameter         | Explanation                                      | Default                 |
| ----------------- | ------------------------------------------------ | ----------------------- |
| `terms`           | Terms patterns. Expects a dictionary.            | `None` (use regex only) |
| `regex`           | RegExp patterns. Expects a dictionary.           | `None` (use terms only) |
| `attr`            | spaCy attribute to match on (eg `NORM`, `LOWER`) | `"TEXT"`                |
| `ignore_excluded` | Whether to skip excluded tokens during matching  | `False`                 |

Patterns, be they `terms` or `regex`, are defined as dictionaries where keys become the label of the extracted entities. Dictionary values are a either a single expression or a list of expressions that match the concept (see [example](#usage)).

## Authors and citation

The `eds.matcher` pipeline was developed by AP-HP's Data Science team.
