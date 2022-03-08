# Getting started

EDS-NLP provides a set of spaCy components that are developed and used to extract information from clinical notes written in French.

If it's your first time with spaCy, we recommend you familiarise yourself with some of their key concepts by looking at the "spaCy 101" page.

## Quick start

### Installation

You can install EDS-NLP via `pip`:

<!-- termynal -->

```
$ pip install edsnlp
---> 100%
Installed
```

We recommend pinning the library version in your projects, or use a strict package manager like [Poetry](https://python-poetry.org/).

```
pip install edsnlp==0.4.0
```

### A first pipeline

Once you've installed the library, let's begin with a very simple example that extracts mentions of COVID19 in a text, and detects whether they are negated.

```python
import spacy

nlp = spacy.blank("fr")  # (1)

terms = dict(
    covid=["covid", "coronavirus"],  # (2)
)

# Sentencizer component, needed for negation detection
nlp.add_pipe("eds.sentences")  # (3)
# Matcher component
nlp.add_pipe("eds.matcher", config=dict(terms=terms))  # (4)
# Negation detection
nlp.add_pipe("eds.negation")

# Process your text in one call !
doc = nlp("Le patient est atteint de covid")

doc.ents  # (5)
# Out: (covid,)

doc.ents[0]._.negation  # (6)
# Out: False
```

1. We only need spaCy's French tokenizer.
1. This example terminology provides a very simple, and by no means exhaustive, list of synonyms for COVID19.
1. In spaCy, pipelines are added via the [`nlp.add_pipe` method](https://spacy.io/api/language#add_pipe). EDS-NLP pipelines are automatically discovered by spaCy.
1. See the [matching tutorial](home/tutorials/matching-a-terminology.md) for mode details.
1. spaCy stores extracted entities in the [`Doc.ents` attribute](https://spacy.io/api/doc#ents).
1. The [`eds.negation` pipeline](pipelines/qualifiers/negation.md) has added a `negation` custom attribute.

This example is complete, it should run as-is. Check out the [spaCy 101 page](home/spacy101.md) if you're not familiar with spaCy.

## Available pipeline components

=== "Core"

    | Pipeline               | Description                                     |
    | ---------------------- | ----------------------------------------------- |
    | `eds.normalizer`       | Non-destructive input text normalisation        |
    | `eds.sentences`        | Better sentence boundary detection              |
    | `eds.matcher`          | A simple yet powerful entity extractor          |
    | `eds.advanced-matcher` | A conditional entity extractor                  |
    | `eds.endlines`         | An unsupervised model to classify each end line |

=== "Qualifiers"

    | Pipeline               | Description                                     |
    | ---------------------- | ----------------------------------------------- |
    | `eds.negation`         | Rule-based negation detection                   |
    | `eds.family`           | Rule-based family context detection             |
    | `eds.hypothesis`       | Rule-based speculation detection                |
    | `eds.reported_speech`  | Rule-based reported speech detection            |
    | `eds.history`          | Rule-based medical history detection            |

=== "Miscellaneous"

    | Pipeline               | Description                                     |
    | ---------------------- | ----------------------------------------------- |
    | `eds.dates`            | Date extraction and normalisation               |
    | `eds.sections`         | Section detection                               |
    | `eds.reason`           | Rule-based hospitalisation reason detection     |

=== "NER"

    | Pipeline                 | Description                |
    | ------------------------ | -------------------------- |
    | `eds.charlson`           | A Charlson score extractor |
    | `eds.sofa`               | A SOFA score extractor     |
    | `eds.emergency.priority` | A priority score extractor |
    | `eds.emergency.ccmu`     | A CCMU score extractor     |
    | `eds.emergency.gemsa`    | A GEMSA score extractor    |

## Disclaimer

The performances of an extraction pipeline may depend on the population and documents that are considered.

## Contributing to EDS-NLP

We welcome contributions ! Fork the project and propose a pull request. Take a look at the [dedicated page](development/contributing.md) for detail.
