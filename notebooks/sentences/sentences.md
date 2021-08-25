---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
%reload_ext autoreload
%autoreload 2
```

```python
# Importation du "contexte", ie la bibliothèque sans installation
import context
```

```python
import spacy
```

```python
from edsnlp.rules.sentences import SentenceSegmenter
```

# Sentences

```python
import re
import spacy

from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex

# Ajout de règles supplémentaires pour gérer les infix
def custom_tokenizer(nlp):
    infix_re = re.compile(r'''[\,\?\:\;\‘\’\`\“\”\"\'~/\(\)\.\+=(->)\$]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes + ['-'])
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)
    return Tokenizer(
        nlp.vocab,
        prefix_search=prefix_re.search,
        suffix_search=suffix_re.search,
        infix_finditer=infix_re.finditer,
    )

def new_nlp():

    nlp = spacy.blank('fr')
    nlp.tokenizer = custom_tokenizer(nlp)

    return nlp
```

```python
text = (
    "Le patient est admis pour des douleurs dans le bras droit, mais n'a pas de problème de locomotion. "
    "Historique d'AVC dans la famille mais\n"
    "pourrait être un cas de rhume\n"
    "Pourrait aussi être un cas de rhume.\n"
    "Motif :\n"
    "-problème de locomotions\n"
    "Douleurs dans le bras droit.\n\n\n\n"
)
```

```python
nlp = new_nlp()
nlp.add_pipe('sentencizer')
```

```python
doc = nlp(text)
```

```python
for sent in doc.sents:
    print('##', repr(sent.text))
```

```python
nlp = new_nlp()
```

```python
sentencer = SentenceSegmenter()
```

```python
doc = sentencer(nlp(text))
```

```python
for sent in doc.sents:
    print('##', repr(sent.text))
```

Note that the newline character is now linked to the preceding sentence. That is especially relevant if the note ends on a newline.

```python

```
