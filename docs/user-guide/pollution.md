# Removing pollution

## Non-destruction

All text normalisation in NLP Tools is non-destructive, ie

```python
nlp(text).text == text
```

is always true.

Hence, the strategy chosen for the pollution pipeline is the following:
1. Tag, **but do not remove**, pollutions on the `Token._.pollution` extension.
2. Propose a `Doc._.clean_` extension, to retrieve the cleaned text.


## Recipes

```python
import spacy
from nlptools import components

nlp = spacy.blank('fr')
nlp.add_pipe('pollution')  # exposed via nlptools.components

text = (
    "Le patient est admis pour des douleurs dans le bras droit, mais n'a pas de problème de locomotion. "
    "NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNBWbWbNbNbNBWbNbWbNbWBNbNbWbNbNBNbWbWbNbWBNbNbWbNBNbWbWbNb\n"
    "Pourrait être un cas de rhume.\n"
    "Motif :\n"
    "Douleurs dans le bras droit."
)

doc = nlp(text)
```


## Working on the cleaned text

Should you need to implement a pipeline using the cleaned version of the documents, the Pollution pipeline also exposes a `Doc._.clean_char_span` method to realign annotations made on the clean text with the original document.

```python
clean = nlp(doc._.clean)
span = clean[27:28]

doc._.clean_[span.start_char:span.end_char]
# Out: 'rhume'

doc.text[span.start_char:span.end_char]
# Out: 'bWbNb'

doc._.char_clean_span(span.start_char, span.end_char)
# Out: rhume
```
