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

nlp.add_pipe(
    "eds.matcher",
    config=dict(
        terms=terms,
        regex=regex,
        attr="LOWER",
        term_matcher="exact",
        term_matcher_config={},
    ),
)
```

1. Every key in the `terms` dictionary is mapped to a concept.
2. The `eds.matcher` pipeline expects a list of expressions, or a single expression.
3. We can also define regular expression patterns.

This snippet is complete, and should run as is.

## Configuration

The pipeline can be configured using the following parameters :

::: edsnlp.pipelines.core.matcher.factory.create_component
    options:
        only_parameters: true

Patterns, be they `terms` or `regex`, are defined as dictionaries where keys become the label of the extracted entities. Dictionary values are a either a single expression or a list of expressions that match the concept (see [example](#usage)).

## Authors and citation

The `eds.matcher` pipeline was developed by AP-HP's Data Science team.
