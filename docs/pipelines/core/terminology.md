# Terminology

EDS-NLP simplifies the terminology matching process by exposing a `eds.terminology` pipeline
that can match on terms or regular expressions.

The terminology matcher is very similar to the [generic matcher](matcher.md), although the use case differs slightly.
The generic matcher is designed to extract any entity, while the terminology matcher is specifically tailored
towards high volume terminologies.

There are some key differences:

1. It labels every matched entity to the same value, provided to the pipeline.
2. The keys provided in the `regex` and `terms` dictionaries are used as the `kb_id_` of the entity.

For instance, a terminology matcher could detect every drug mention under the top-level label `drug`,
and link each individual mention to a given drug through its `kb_id_` attribute.

## Usage

Let us redefine the pipeline :

```python
import spacy

nlp = spacy.blank("fr")

terms = dict(
    covid=["coronavirus", "covid19"],  # (1)
    flu=["grippe saisonnière"],  # (2)
)

regex = dict(
    covid=r"coronavirus|covid[-\s]?19|sars[-\s]cov[-\s]2",  # (3)
)

nlp.add_pipe(
    "eds.terminology",
    config=dict(
        label="disease",
        terms=terms,
        regex=regex,
        attr="LOWER",
    ),
)
```

1. Every key in the `terms` dictionary is mapped to a concept.
2. The `eds.matcher` pipeline expects a list of expressions, or a single expression.
3. We can also define regular expression patterns.

This snippet is complete, and should run as is.

## Configuration

The pipeline can be configured using the following parameters :

| Parameter         | Explanation                                      | Default                 |
| ----------------- | ------------------------------------------------ | ----------------------- |
| `label`           | Top-level label.                                 | Required                |
| `terms`           | Terms patterns. Expects a dictionary.            | `None` (use regex only) |
| `regex`           | RegExp patterns. Expects a dictionary.           | `None` (use terms only) |
| `attr`            | spaCy attribute to match on (eg `NORM`, `LOWER`) | `"TEXT"`                |
| `ignore_excluded` | Whether to skip excluded tokens during matching  | `False`                 |

Patterns, be they `terms` or `regex`, are defined as dictionaries where keys become the `kb_id_` of the extracted entities.
Dictionary values are a either a single expression or a list of expressions that match the concept (see [example](#usage)).

## Authors and citation

The `eds.terminology` pipeline was developed by AP-HP's Data Science team.
