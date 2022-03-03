# Overview

EDS-NLP provides a set of SpaCy components that are developed and used at AP-HP[^1].

If it's your first time with SpaCy, we recommend you familiarise yourself with some of their key concepts by looking at their "SpaCy 101" page.

## Quick start

Once you've [installed the library](home/installation.md), let's begin with a very simple example that extracts mentions of COVID in a text, and detects whether they are negated.

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

1. We only need SpaCy's French tokenizer.
1. This example terminology provides a very simple, and by no means exhaustive, list of synonyms for COVID.
1. In SpaCy, pipelines are added via the [`nlp.add_pipe` method](https://spacy.io/api/language#add_pipe). EDS-NLP pipelines are automatically discovered by SpaCy.
1. See the [matching tutorial](home/tutorials/matching-a-terminology.md) for mode details.
1. Spacy keeps extracted entities in the [`Doc.ents` attribute](https://spacy.io/api/doc#ents).
1. The [`eds.negation` pipeline](pipelines/qualifiers/negation.md) has added a `negation` custom attribute.

This example is complete, it should run as-is. Check out the [SpaCy 101 page](home/spacy101.md) if you're not familiar with SpaCy.

## Available pipelines

=== "Core"

    | Pipeline               | Description                                     |
    | ---------------------- | ----------------------------------------------- |
    | `eds.normalizer`       | Non-destructive input text normalization        |
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
    | `eds.antecedent`       | Rule-based antecedent detection                 |

=== "Miscellaneous"

    | Pipeline               | Description                                     |
    | ---------------------- | ----------------------------------------------- |
    | `eds.dates`            | Date extraction and normalization               |
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

You should properly validate your pipelines before deploying them. Some (but not all) components from EDS-NLP underwent some form of validation, but the performance varies and you should always verify the results on your own data.

We recommend using [EDS-LabelTool](https://gitlab.eds.aphp.fr/datasciencetools/labeltool) to validate your pipelines. EDS-LabelTool enables quick and easy annotation from the notebook.

## Contributing to EDS-NLP

We welcome contributions ! Fork the project and propose a pull request. Take a look at the [dedicated page](development/contributing.md) for detail.

[^1]:
    **Assistance Publique - Hôpitaux de Paris**, or Greater Paris University Hospital,
    is a group of 39 public hospitals in the Greater Paris area.
