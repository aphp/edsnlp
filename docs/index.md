# Documentation for EDS-NLP

```{eval-rst}
.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Getting Started

    getting-started/installation
    getting-started/architecture

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: User Guide

    user-guide/normalisation
    user-guide/sentences
    user-guide/matcher
    user-guide/negation
    user-guide/family
    user-guide/hypothesis
    user-guide/reported-speech
    user-guide/antecedents
    user-guide/sections
    user-guide/dates
    user-guide/consultation-dates
    user-guide/score
    user-guide/endlines
    user-guide/reason

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Connectors

    connectors/omop
    connectors/brat
    connectors/labeltool
    connectors/spark

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: List of functions

    api/base
    api/pipelines
    api/qualifiers
    api/connectors
    api/utilities

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Tutorials

    tutorials/first-pipeline
    tutorials/getting-faster
    tutorials/notebooks
    tutorials/word-vectors
    tutorials/endlines-example
    tutorials/working-with-spark
    tutorials/reason-example

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Additional information

    additional/contributing
    additional/changelog
```

EDS-NLP provides a set of Spacy components that are used at AP-HP. We focus on usability and non-destructiveness.

## Running the interactive demo

To get a glimpse of what EDS-NLP can do for you, run the interactive demo !

```shell
# Clone the repo
git clone https://gitlab.eds.aphp.fr/datasciencetools/edsnlp.git

# Move to the repo directory
cd edsnlp

# Install the project with the demo requirements
pip install '.[demo]'

# Run the demo
streamlit run scripts/demo.py
```

Go to the provided URL to see the library in action.

```{warning}
The above code will not work within JupyterLab. You need to execute it locally.
```

## Quick start

Let us begin with a very simple example that extracts mentions of COVID in a text, and detects whether they are negated.

```python
import spacy

# Load declared pipelines
from edsnlp import components

nlp = spacy.blank("fr")

terms = dict(
    covid=["covid", "coronavirus"],
)

# Sentencizer component, needed for negation detection
nlp.add_pipe("sentences")
# Matcher component
nlp.add_pipe("matcher", config=dict(terms=terms))
# Negation detection
nlp.add_pipe("negation")

# Process your text in one call !
doc = nlp("Le patient est atteint de covid")

doc.ents
# Out: (covid,)

doc.ents[0]._.negation
# Out: False
```

This example is complete, it should run as-is.

## Available pipelines

| Pipeline              | Description                                     |
| --------------------- | ----------------------------------------------- |
| `eds.normalizer`      | Non-destructive input text normalization        |
| `eds.sentences`       | Better sentence boundary detection              |
| `eds.matcher`         | A simple yet powerful entity extractor          |
| `eds.negation`        | Rule-based negation detection                   |
| `eds.family`          | Rule-based family context detection             |
| `eds.hypothesis`      | Rule-based speculation detection                |
| `eds.antecedent`      | Rule-based antecedent detection                 |
| `eds.reported_speech` | Rule-based reported speech detection            |
| `eds.sections`        | Section detection                               |
| `eds.dates`           | Date extraction and normalization               |
| `eds.score`           | A simple clinical score extractor               |
| `eds.endlines`        | An unsupervised model to classify each end line |
| `eds.reason`          | Rule-based hospitalisation reason detection     |

## Disclaimer

EDS-NLP is still young and in constant evolution. Although we strive to remain backward-compatible, the API can be subject to breaking changes. Moreover, you should properly validate your pipelines before deploying them. Some (but not all) components from EDS-NLP underwent some form of validation, but the performance varies and you should always verify the results on your own data.

We recommend using [EDS-LabelTool](https://gitlab.eds.aphp.fr/datasciencetools/labeltool) to validate your pipelines. EDS-LabelTool enables quick and easy annotation from the notebook.

## Contributing to EDS-NLP

We welcome contributions ! Fork the project and propose a pull request. Take a look at the [dedicated page](additional/contributing.md) for detail.
