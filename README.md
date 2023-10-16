![Tests](https://img.shields.io/github/actions/workflow/status/aphp/edsnlp/tests.yml?branch=master&label=tests&style=flat-square)
[![Documentation](https://img.shields.io/github/actions/workflow/status/aphp/edsnlp/documentation.yml?branch=master&label=docs&style=flat-square)](https://aphp.github.io/edsnlp/latest/)
[![PyPI](https://img.shields.io/pypi/v/edsnlp?color=blue&style=flat-square)](https://pypi.org/project/edsnlp/)
[![Demo](https://img.shields.io/badge/demo%20%F0%9F%9A%80-streamit-grean?style=flat-square)](https://aphp.github.io/edsnlp/demo/)
[![Codecov](https://img.shields.io/codecov/c/github/aphp/edsnlp?logo=codecov&style=flat-square)](https://codecov.io/gh/aphp/edsnlp)
[![DOI](https://zenodo.org/badge/467585436.svg)](https://zenodo.org/badge/latestdoi/467585436)


EDS-NLP
=======

EDS-NLP is a collaborative NLP framework that aims at extracting information from French clinical notes.
At its core, it is a collection of components or pipes, either rule-based functions or
deep learning modules. These components are organized into a novel efficient and modular pipeline system, built for hybrid and multitask models. We use [spaCy](https://spacy.io) to represent documents and their annotations, and [Pytorch](https://pytorch.org/) as a deep-learning backend for trainable components.

EDS-NLP is versatile and can be used on any textual document. The rule-based components are fully compatible with spaCy's pipelines, and vice versa. This library is a product of collaborative effort, and we encourage further contributions to enhance its capabilities.

Check out our interactive [demo](https://aphp.github.io/edsnlp/demo/) !

## Quick start

### Installation

You can install EDS-NLP via `pip`. We recommend pinning the library version in your projects, or use a strict package manager like [Poetry](https://python-poetry.org/).

```shell
pip install edsnlp==0.10.0beta1
```

or if you want to use the trainable components (using pytorch)

```shell
pip install "edsnlp[ml]==0.10.0beta1"
```

### A first pipeline

Once you've installed the library, let's begin with a very simple example that extracts mentions of COVID19 in a text, and detects whether they are negated.

```python
import edsnlp

nlp = edsnlp.blank("eds")

terms = dict(
    covid=["covid", "coronavirus"],
)

# Split the documents into sentences, this isneeded for negation detection
nlp.add_pipe("eds.sentences")
# Matcher component
nlp.add_pipe("eds.matcher", config=dict(terms=terms))
# Negation detection
nlp.add_pipe("eds.negation")

# Process your text in one call !
doc = nlp("Le patient n'est pas atteint de covid")

doc.ents
# Out: (covid,)

doc.ents[0]._.negation
# Out: True
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
  author = {Wajsburt, Perceval and Petit-Jean, Thomas and Dura, Basile and Cohen, Ariel and Jean, Charline and Bey, Romain},
  doi    = {10.5281/zenodo.6424993},
  title  = {EDS-NLP: efficient information extraction from French clinical notes},
  url    = {https://aphp.github.io/edsnlp}
}
```

## Acknowledgement

We would like to thank [Assistance Publique – Hôpitaux de Paris](https://www.aphp.fr/), [AP-HP Foundation](https://fondationrechercheaphp.fr/) and [Inria](https://www.inria.fr) for funding this project.
