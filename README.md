# EDS-NLP

A simple library to group together the different pre-processing pipelines that are used at AP-HP, as SpaCy components. We focus on **usability and non-destructiveness**.

## Getting started

### Installation

Installation is straightforward. To get the latest version :

```
pip install git+https://gitlab.eds.aphp.fr/datasciencetools/edsnlp.git
```

We recommend pinning the version of the library :

```
pip install git+https://gitlab.eds.aphp.fr/datasciencetools/edsnlp.git@v0.4.0
```

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

### Quick start

Let us begin with a very simple example that extracts mentions of COVID in a text, and detects whether they are negated.

```python
import spacy

# Load declared pipelines


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

This example is complete, it should run as-is. See the [documentation](https://datasciencetools-pages.eds.aphp.fr/edsnlp/documentation) for detail.

### Available pipelines

| Pipeline              | Description                            |
| --------------------- | -------------------------------------- |
| `eds.normalizer`      | Non-destructive text normalization     |
| `eds.sentences`       | Better sentence boundary detection     |
| `eds.matcher`         | A simple yet powerful entity extractor |
| `eds.negation`        | Rule-based negation detection          |
| `eds.family`          | Rule-based family context detection    |
| `eds.hypothesis`      | Rule-based speculation detection       |
| `eds.antecedent`      | Rule-based antecedent detection        |
| `eds.reported_speech` | Rule-based reported speech detection   |
| `eds.sections`        | Section detection                      |
| `eds.dates`           | Date extraction and normalization      |
| `eds.score`           | A simple clinical score extractor      |

## Disclaimer

EDS-NLP is still young and in constant evolution. Although we strive to remain backward-compatible, the API can be subject to breaking changes. Moreover, you should properly validate your pipelines before deploying them. Some (but not all) components from EDS-NLP underwent some form of validation, but you should nonetheless always verify the results on your own data.

## Contributing to EDS-NLP

We welcome contributions ! Fork the project and propose a pull request. Take a look at the [dedicated page](https://datasciencetools-pages.eds.aphp.fr/edsnlp/documentation/additional/contributing.html) for detail.
