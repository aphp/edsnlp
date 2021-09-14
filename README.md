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

### Quick start

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

See the [documentation](https://equipedatascience-pages.eds.aphp.fr/edsnlp/) for detail.

## Disclaimer

EDS-NLP is still young and in constant evolution. Although we strive to remain backward-compatible, the API can be subject to breaking changes. Moreover, you should properly validate your pipelines before deploying them. Some (but not all) components from EDS-NLP underwent some form of validation, but you should nonetheless always verify the results on your own data.

## Contributing to EDS-NLP

We welcome contributions ! Fork the project and propose a pull request. Take a look at the [dedicated page](https://equipedatascience-pages.eds.aphp.fr/edsnlp/additional/contributing.html) for detail.
