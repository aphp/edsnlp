# EDS-NLP

EDS-NLP provides a set of spaCy components that are used to extract information from clinical notes written in French.

If it's your first time with spaCy, we recommend you familiarise yourself with some of their key concepts by looking at the "spaCy 101" page.

## Quick start

### Installation

You can install EDS-NLP via `pip`:

```shell
python -m pip install edsnlp
```

We recommend pinning the library version in your projects, or use a strict package manager like [Poetry](https://python-poetry.org/).

```shell
python -m pip install edsnlp==0.4.0
```

### A first pipeline

Once you've installed the library, let's begin with a very simple example that extracts mentions of COVID19 in a text, and detects whether they are negated.

```python
import spacy

nlp = spacy.blank("fr")

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

Go to the [documentation](https://aphp.github.io/edsnlp) for more information!

## Disclaimer

The performances of an extraction pipeline may depend on the population and documents that are considered.

## Contributing to EDS-NLP

We welcome contributions ! Fork the project and propose a pull request. Take a look at the [dedicated page](development/contributing.md) for detail.
