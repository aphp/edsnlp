# EDS-NLP

A simple library to group together the different pre-processing pipelines that are used at AP-HP, as Spacy components. We focus on **usability and non-destructiveness**.

## Getting started

### Installation

Installation is straightforward. To get the latest version :

```
pip install git+https://gitlab.eds.aphp.fr/equipedatascience/edsnlp.git
```

We recommand pinning the version of the library :

```
pip install git+https://gitlab.eds.aphp.fr/equipedatascience/edsnlp.git@v0.2.0
```

### Available pipelines

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

Check out the [documentation](https://equipedatascience-pages.eds.aphp.fr/edsnlp) for more detail.

### Creating a pipeline

```python
import spacy

# Load declared pipelines
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("sections")
```

To declare an entity matcher:

```python
terms = dict(
    covid=["covid", "coronavirus"],
)

nlp.add_pipe("matcher", config=dict(terms=terms))
```

See the [documentation](https://equipedatascience-pages.eds.aphp.fr/edsnlp/) for detail.

## Contributing to EDS-NLP

We welcome contributions ! Fork the project and propose a pull request. Take a look at the [dedicated page](https://equipedatascience-pages.eds.aphp.fr/edsnlp/additional/contributing.html) for detail.
