# Documentation for EDS-NLP

```{eval-rst}
.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Getting Started

    getting-started/installation
    getting-started/quickstart
    getting-started/architecture

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: User Guide

    user-guide/normalizer
    user-guide/sentences
    user-guide/matcher
    user-guide/negation
    user-guide/family
    user-guide/hypothesis
    user-guide/reported-speech
    user-guide/antecedents
    user-guide/pollution
    user-guide/sections
    user-guide/dates
    user-guide/quickumls

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Connectors

    connectors/brat
    connectors/labeltool

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: API

    api/base
    api/pipelines
    api/connectors
    api/utilities

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Tutorials

    tutorials/first-pipeline
    tutorials/notebooks
    tutorials/word-vectors

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Additional information

    additional/contributing
    additional/changelog
```

EDS-NLP provides a set of Spacy components that are used at AP-HP. We focus on usability and non-destructiveness.

## Available pipelines

| Pipeline     | Description                                                           |
| ------------ | --------------------------------------------------------------------- |
| `normalizer` | Non-destructive input text normalization                              |
| `sentences`  | Better sentence boundary detection                                    |
| `matcher`    | A simple yet powerful entity extractor                                |
| `negation`   | Rule-based negation detection                                         |
| `family`     | Rule-based family context detection                                   |
| `hypothesis` | Rule-based speculation detection                                      |
| `antecedent` | Rule-based antecedent detection                                       |
| `rspeech`    | Rule-based reported speech detection                                  |
| `sections`   | Section detection                                                     |
| `pollution`  | Pollution detection and non-destructive removal                       |
| `dates`      | Date extraction and normalization                                     |
| `quickumls`  | A basic Spacy v3 re-implementation of Georgetown's QuickUMLS pipeline |

## Quick start

The following example is complete, it should run as-is.

```python
import spacy

# Load declared pipelines
from edsnlp import components

nlp = spacy.blank("fr")

terms = dict(
    covid=["covid", "coronavirus"],
)

nlp.add_pipe("matcher", config=dict(terms=terms))

doc = nlp("Le patient est atteint de covid")
doc.ents
# Out: (covid,)
```

## Disclaimer

EDS-NLP is still young and in constant evolution. Although we strive to remain backward-compatible, the API can be subject to breaking changes. Moreover, you should properly validate your pipelines before deploying them. Some (but not all) components from EDS-NLP underwent some form of validation, but the performance varies and you should always verify the results on your own data.

We recommend using [EDS-LabelTool](https://gitlab.eds.aphp.fr/datasciencetools/labeltool) to validate your pipelines. EDS-LabelTool enables quick and easy annotation from the notebook.

## Contributing to EDS-NLP

We welcome contributions ! Fork the project and propose a pull request. Take a look at the [dedicated page](additional/contributing.md) for detail.
