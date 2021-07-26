# LabelTool Connector

[LabelTool](https://gitlab.eds.aphp.fr/datasciencetools/labeltool) is an in-house module enabling rapid annotation of pre-extracted entities.

We provide a ready-to-use function that converts a list of annotated Spacy documents into a `pandas` DataFrame that is readable to LabelTool.

```python
import spacy
from edsnlp import components
from negparhyp import components
from edsnlp.utils.labeltool import docs2labeltool

corpus = [
    "Ceci est un document médical.",
    "Le patient n'est pas malade.",
]

# Instantiate the spacy pipeline
nlp = spacy.blank('fr')
nlp.add_pipe('sentences')
nlp.add_pipe('matcher', config=dict(terms=dict(medical='médical', malade='malade')))
nlp.add_pipe('negation')

# Convert all BRAT files to a list of documents
docs = nlp.pipe(corpus)

df = docs2labeltool(docs, extensions=['negated'])

print(df)
#    note_id                      note_text  offset_begin  offset_end    label  \
# 0        0  Ceci est un document médical.            21          28  medical   
# 1        1   Le patient n'est pas malade.            21          27   malade   
# 
#   lexical_variant  negated  
# 0         médical    False  
# 1          malade     True  
```
