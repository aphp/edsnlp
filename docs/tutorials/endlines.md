# Detecting end-of-lines

A common problem in medical corpus is that the character `\n` does not necessarily correspond to a real new line as in other domains.

For example, it is common to find texts like:

```
Il doit prendre
le medicament indiqué 3 fois par jour. Revoir médecin
dans 1 mois.
```

!!! note "Inserted new line characters"

    This issue is especially impactful for clinical notes that have been extracted from PDF documents.
    In that case, the new line character could be deliberately inserted by the doctor, or more likely
    added to respect the layout during the edition of the PDF.

The aim of this tutorial is to train a unsupervised model to detect this _false endlines_ and to use it for inference.
The implemented model is based on the work of Zweigenbaum et al[@zweigenbaum2016].

## Training the model

Let's train the model using an example corpus of three documents:

```python
import spacy
from edsnlp.pipelines.core.endlines import EndLinesModel

nlp = spacy.blank("fr")

text1 = """Le patient est arrivé hier soir.
Il est accompagné par son fils

ANTECEDENTS
Il a fait une TS en 2010;
Fumeur, il est arrêté il a 5 mois
Chirurgie de coeur en 2011
CONCLUSION
Il doit prendre
le medicament indiqué 3 fois par jour. Revoir médecin
dans 1 mois.
DIAGNOSTIC :

Antecedents Familiaux:
- 1. Père avec diabète
"""

text2 = """J'aime le \nfromage...\n"""
text3 = (
    "/n"
    "Intervention(s) - acte(s) réalisé(s) :/n"
    "Parathyroïdectomie élective le [DATE]"
)

texts = [
    text1,
    text2,
    text3,
]

corpus = nlp.pipe(texts)

# Fit the model
endlines = EndLinesModel(nlp=nlp)  # (1)
df = endlines.fit_and_predict(corpus)  # (2)

# Save model
PATH = "/tmp/path_to_model"
endlines.save(PATH)
```

1. Initialize the [`EndLinesModel`][edsnlp.pipelines.core.endlines.endlinesmodel.EndLinesModel]
   object and then fit (and predict) in the training corpus.
2. The corpus should be an iterable of spacy documents.

## Use a trained model for inference

<!-- no-check -->

```python
import spacy

nlp = spacy.blank("fr")

PATH = "/path_to_model"
nlp.add_pipe("eds.endlines", config=dict(model_path=PATH))  # (1)
nlp.add_pipe("eds.sentences")  # (1)

docs = list(nlp.pipe([text1, text2, text3]))

doc = docs[1]
doc
# Out: J'aime le
# Out: fromage...

list(doc.sents)[0]
# Out: J'aime le
# Out: fromage...
```

1. You should specify the path to the trained model here.
2. All fake new line are excluded by setting their `tag` to 'EXCLUDED' and all true new lines' `tag` are set to 'ENDLINE'.

## Declared extensions

It lets downstream matchers skip excluded tokens (see [normalisation](../pipelines/core/normalisation.md)) for more detail.

\bibliography
