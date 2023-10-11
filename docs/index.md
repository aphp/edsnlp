# Getting started

EDS-NLP is a collaborative NLP framework that aims at extracting information from French clinical notes.
At its core, it is a collection of components or pipes, either rule-based functions or
[deep learning modules](https://aphp.github.io/concepts/torch-component). These components are organized into a novel efficient and modular [pipeline system](https://aphp.github.io/concepts/pipeline), built for hybrid and multi-task models. We use [spaCy](https://spacy.io) to represent documents and their annotations, and [Pytorch](https://pytorch.org/) as a deep-learning backend for trainable components.

Although initially designed for French clinical notes, the architecture of EDS-NLP is versatile and can be used on any document. The rule-based components are fully compatible with spaCy's pipelines, and vice versa, which makes it easy to integrate and extend with other NLP tools. This library is a product of collaborative effort, and we encourage further contributions to enhance its capabilities. Check out our interactive [demo](https://aphp.github.io/edsnlp/demo/) to see EDS-NLP in action.

## Quick start

### Installation

You can install EDS-NLP via `pip`:

<div class="termy">

```console
$ pip install edsnlp
---> 100%
color:green Successfully installed!
```

</div>

We recommend pinning the library version in your projects, or use a strict package manager like [Poetry](https://python-poetry.org/).

```
pip install edsnlp==0.9.1
```

### A first pipeline

Once you've installed the library, let's begin with a very simple example that extracts mentions of COVID19 in a text, and detects whether they are negated.

```python
import edsnlp

nlp = edsnlp.blank("eds")  # (1)

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
doc = nlp("Le patient n'est pas atteint de covid")

doc.ents  # (5)
# Out: (covid,)

doc.ents[0]._.negation  # (6)
# Out: True
```

1. 'eds' is the name of the language, which defines the [tokenizer](/tokenizers).
2. This example terminology provides a very simple, and by no means exhaustive, list of synonyms for COVID19.
3. In spaCy, pipelines are added via the [`nlp.add_pipe` method](https://spacy.io/api/language#add_pipe). EDS-NLP pipelines are automatically discovered by spaCy.
4. See the [matching tutorial](tutorials/matching-a-terminology.md) for mode details.
5. spaCy stores extracted entities in the [`Doc.ents` attribute](https://spacy.io/api/doc#ents).
6. The `eds.negation` component has adds a `negation` custom attribute.

This example is complete, it should run as-is.

## Tutorials

To learn more about EDS-NLP, we have prepared a series of tutorials that should cover the main features of the library.

--8<-- "docs/tutorials/overview.md:tutorials"

## Available pipeline components

--8<-- "docs/pipelines/overview.md:components"

## Disclaimer

The performances of an extraction pipeline may depend on the population and documents that are considered.

## Contributing to EDS-NLP

We welcome contributions ! Fork the project and propose a pull request. Take a look at the [dedicated page](contributing.md) for detail.

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
