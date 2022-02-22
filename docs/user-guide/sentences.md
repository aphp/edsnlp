# Sentences

The `eds.sentences` pipeline provides an alternative to Spacy's default `sentencizer`, aiming to overcome some of its limitations.

Indeed, the `sentencizer` merely looks at period characters to detect the end of a sentence, a strategy that often fails in a medical note settings. Our `sentences` component also classifies end-of-lines as sentence boundaries if the subsequent token begins with an uppercase character, leading to slightly better performances.

Moreover, the `eds.sentences` pipeline can use the output of the `eds.normalizer` pipeline, and more specifically the end-of-line classification. This is activated by default.

## Usage

```python
import spacy
from edsnlp import components

text = (
    "Le patient est admis le 23 août 2021 pour une douleur à l'estomac\n"
    "Il lui était arrivé la même chose il y a deux ans."
)

# Using Spacy's default sentencizer
nlp = spacy.blank("fr")
nlp.add_pipe("sentencizer")

doc = nlp(text)

for sentence in doc.sents:
    print("<s>", sentence, "</s>")
# Out: <s> Le patient est admis le 23 août 2021 pour une douleur à l'estomac
# Out: Il lui était arrivé la même chose il y a deux ans. <\s>

# Using EDS-NLP's sentences
nlp = spacy.blank("fr")
nlp.add_pipe("eds.sentences")  # exposed via edsnlp.components

doc = nlp(text)

for sentence in doc.sents:
    print("<s>", sentence, "</s>")
# Out: <s> Le patient est admis le 23 août 2021 pour une douleur à l'estomac
# Out:  <\s>
# Out: <s> Il lui était arrivé la même chose il y a deux ans. <\s>
```

## Authors and citation

The `eds.sentences` pipeline was developed at the Data and Innovation unit, IT department, AP-HP.
