![Tests](https://img.shields.io/github/actions/workflow/status/aphp/edsnlp/tests.yml?branch=master&label=tests&style=flat-square)
[![Documentation](https://img.shields.io/github/actions/workflow/status/aphp/edsnlp/documentation.yml?branch=master&label=docs&style=flat-square)](https://aphp.github.io/edsnlp/latest/)
[![PyPI](https://img.shields.io/pypi/v/edsnlp?color=blue&style=flat-square)](https://pypi.org/project/edsnlp/)
[![Demo](https://img.shields.io/badge/demo%20%F0%9F%9A%80-streamit-grean?style=flat-square)](https://aphp.github.io/edsnlp/demo/)
[![Codecov](https://img.shields.io/codecov/c/github/aphp/edsnlp?logo=codecov&style=flat-square)](https://codecov.io/gh/aphp/edsnlp)
[![DOI](https://zenodo.org/badge/467585436.svg)](https://zenodo.org/badge/latestdoi/467585436)

# EDS-NLP

EDS-NLP provides a set of spaCy components that are used to extract information from clinical notes written in French.

Check out the interactive [demo](https://aphp.github.io/edsnlp/demo/)!

If it's your first time with spaCy, we recommend you familiarise yourself with some of their key concepts by looking at the "[spaCy 101](https://aphp.github.io/edsnlp/latest/tutorials/spacy101/)" page in the documentation.

## Quick start

### Installation

You can install EDS-NLP via `pip`:

```shell
pip install edsnlp
```

We recommend pinning the library version in your projects, or use a strict package manager like [Poetry](https://python-poetry.org/).

```shell
pip install edsnlp==0.9.0
```

### A first pipeline

Once you've installed the library, let's begin with a very simple example that extracts mentions of COVID19 in a text, and detects whether they are negated.

```python
import spacy

nlp = spacy.blank("eds")

terms = dict(
    covid=["covid", "coronavirus"],
)

# Sentencizer component, needed for negation detection
nlp.add_pipe("eds.sentences")
# Matcher component
nlp.add_pipe("eds.matcher", config=dict(terms=terms))
# Negation detection
nlp.add_pipe("eds.negation")

# Process your text in one call !
doc = nlp("Le patient est atteint de covid")

doc.ents
# Out: (covid,)

doc.ents[0]._.negation
# Out: False
```

## Documentation

Go to the [documentation](https://aphp.github.io/edsnlp) for more information.

## Disclaimer

The performances of an extraction pipeline may depend on the population and documents that are considered.

## Contributing to EDS-NLP

We welcome contributions ! Fork the project and propose a pull request.
Take a look at the [dedicated page](https://aphp.github.io/edsnlp/latest/contributing/) for detail.

## Citation

If you use EDS-NLP, please cite us as below.

```bibtex
@misc{edsnlp,
  author = {Dura, Basile and Wajsburt, Perceval and Petit-Jean, Thomas and Cohen, Ariel and Jean, Charline and Bey, Romain},
  doi    = {10.5281/zenodo.6424993},
  title  = {EDS-NLP: efficient information extraction from French clinical notes},
  url    = {http://aphp.github.io/edsnlp}
}
```

## Acknowledgement

We would like to thank [Assistance Publique – Hôpitaux de Paris](https://www.aphp.fr/), [AP-HP Foundation](https://fondationrechercheaphp.fr/) and [Inria](https://www.inria.fr) for funding this project.
