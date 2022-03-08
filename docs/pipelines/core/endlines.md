# Endlines

The `eds.endlines` pipeline classifies newline characters as actual end of lines or mere spaces. In the latter case, the token is removed from the normalised document.

Behind the scenes, it uses a `endlinesmodel` instance, which is an unsupervised algorithm based on the work of Zweigenbaum et al[@zweigenbaum2016].

## Usage

The following example shows a simple usage.

### Training

```python
import spacy
from edsnlp.pipelines.endlines.endlinesmodel import EndLinesModel
import pandas as pd
from spacy import displacy

nlp = spacy.blank("fr")

texts = [
    """Le patient est arrivé hier soir.
Il est accompagné par son fils

ANTECEDENTS
Il a fait une TS en 2010;
Fumeur, il est arreté il a 5 mois
Chirurgie de coeur en 2011
CONCLUSION
Il doit prendre
le medicament indiqué 3 fois par jour. Revoir médecin
dans 1 mois.
DIAGNOSTIC :

Antecedents Familiaux:
- 1. Père avec diabete

""",
    """J'aime le \nfromage...\n""",
]

docs = list(nlp.pipe(texts))

# Train and predict an EndLinesModel
endlines = EndLinesModel(nlp=nlp)

df = endlines.fit_and_predict(docs)
df.head()

PATH = "path_to_save"
endlines.save(PATH)
```

### Inference

```python
import spacy

nlp = spacy.blank("fr")
PATH = "path_to_save"
nlp.add_pipe("eds.endlines", config=dict(model_path=PATH))

docs = list(nlp.pipe(texts))

doc_exemple = docs[1]

doc_exemple

doc_exemple.ents = tuple(
    s for s in doc_exemple.spans["new_lines"] if s.label_ == "space"
)

displacy.render(doc_exemple, style="ent", options={"colors": {"space": "red"}})
```

## Configuration

The pipeline can be configured using the following parameters :

| Parameter    | Explanation                      | Default  |
| ------------ | -------------------------------- | -------- |
| `model_path` | Path to the pre-trained pipeline | Required |

## Declared extensions

The `eds.endlines` pipeline declares one [spaCy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects. The `end_line` attribute is a boolean, set to `True` if the pipeline predicts that the new line is an end line character. Otherwise, it is set to `False` if the new line is classified as a space.

The pipeline also sets the `excluded` custom attribute on newlines that are classified as spaces. It lets downstream matchers skip excluded tokens (see [normalisation](./normalisation.md)) for more detail.

## Authors and citation

The `eds.endlines` pipeline was developed by AP-HP's Data Science team.

\bibliography
