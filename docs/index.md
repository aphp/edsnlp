# Getting started

EDS-NLP is a collaborative NLP framework that aims at extracting information from French clinical notes.
At its core, it is a collection of components or pipes, either rule-based functions or
deep learning modules. These components are organized into a novel efficient and modular pipeline system, built for hybrid and multitask models. We use [spaCy](https://spacy.io) to represent documents and their annotations, and [Pytorch](https://pytorch.org/) as a deep-learning backend for trainable components.

EDS-NLP is versatile and can be used on any textual document. The rule-based components are fully compatible with spaCy's pipelines, and vice versa. This library is a product of collaborative effort, and we encourage further contributions to enhance its capabilities.

Check out our interactive [demo](https://aphp.github.io/edsnlp/demo/) !

## Quick start

### Installation

You can install EDS-NLP via `pip`. We recommend pinning the library version in your projects, or use a strict package manager like [Poetry](https://python-poetry.org/).

```{: data-md-color-scheme="slate" }
pip install edsnlp==0.17.2
```

or if you want to use the trainable components (using pytorch)

```{: data-md-color-scheme="slate" }
pip install "edsnlp[ml]==0.17.2"
```

### A first pipeline

Once you've installed the library, let's begin with a very simple example that extracts mentions of COVID19 in a text, and detects whether they are negated.

```python
import edsnlp, edsnlp.pipes as eds

nlp = edsnlp.blank("eds")  # (1)

terms = dict(
    covid=["covid", "coronavirus"],  # (2)
)

# Sentencizer component, needed for negation detection
nlp.add_pipe(eds.sentences())  # (3)
# Matcher component
nlp.add_pipe(eds.matcher(terms=terms))  # (4)
# Negation detection
nlp.add_pipe(eds.negation())

# Process your text in one call !
doc = nlp("Le patient n'est pas atteint de covid")

doc.ents  # (5)
# Out: (covid,)

doc.ents[0]._.negation  # (6)
# Out: True
```

1. 'eds' is the name of the language, which defines the [tokenizer](/tokenizers).
2. This example terminology provides a very simple, and by no means exhaustive, list of synonyms for COVID19.
3. Similarly to spaCy, pipes are added via the [`nlp.add_pipe` method](https://spacy.io/api/language#add_pipe).
4. See the [matching tutorial](tutorials/matching-a-terminology.md) for mode details.
5. spaCy stores extracted entities in the [`Doc.ents` attribute](https://spacy.io/api/doc#ents).
6. The `eds.negation` component has adds a `negation` custom attribute.

This example is complete, it should run as-is.

## Tutorials

To learn more about EDS-NLP, we have prepared a series of tutorials that should cover the main features of the library.

--8<-- "docs/tutorials/index.md:classic-tutorials"

We also provide tutorials on how to train deep-learning models with EDS-NLP. These tutorials cover the training API, hyperparameter tuning, and more.

--8<-- "docs/tutorials/index.md:deep-learning-tutorials"

## Available pipeline components

--8<-- "docs/pipes/index.md:components"

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
