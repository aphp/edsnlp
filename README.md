# EDS-NLP

A simple library to group together the different pre-processing pipelines that are used at AP-HP, as Spacy components.

We focus on usability and non-destructiveness.


## Getting started

### Installation

We recommend cloning the repository to install the library. That way, you will be able to get started faster thanks to the example notebooks.

```
git clone https://gitlab.eds.aphp.fr/equipedatascience/nlptools.git
pip install ./nlptools
```


### Available pipelines

- [`matcher`](edsnlp/rules/generic.py): a generic matching tool, with RegEx/term and fuzzy matching support.
- [`pollution`](edsnlp/rules/pollution/pollution.py): non-destructive detection of pollutions
- [`sections`](edsnlp/rules/sections/sections.py): detection of section titles and inference of section spans
- [`quickumls`](edsnlp/rules/quickumls/quickumls.py): a basic re-implementation of the spacy component from Georgetown's [QuickUMLS](https://github.com/Georgetown-IR-Lab/QuickUMLS)


### Creating a pipeline

```python
import spacy

# Load declared pipelines
from edsnlp import components

nlp = spacy.blank('fr')
nlp.add_pipe('sections')
```

To declare an entity matcher:

```python
terms = dict(
    covid=['covid', 'coronavirus'],
)

nlp.add_pipe('matcher', config=dict(terms=terms))
```

See the documentation for detail.


## Documentation

The documentation is available [here](https://equipedatascience-pages.eds.aphp.fr/edsnlp/). Due to an issue with Gitlab Pages, you will need to change the address, to put an hyphen in place of the first point :

```diff
- https://projects.pages.eds.aphp.fr/blabla...
+ https://projects-pages.eds.aphp.fr/blabla...
```

The documentation will be available in `docs/_build/html` (open `index.html` in your favorite web browser).
