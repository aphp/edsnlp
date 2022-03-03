---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: "1.3"
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
# One-shot import of all declared SpaCy components

```

# Baselines

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
nlp = new_nlp()
```

```python
# nlp.add_pipe('sentencizer')
nlp.add_pipe('matcher', config=dict(regex=dict(douleurs=['blème de locomotion', 'douleurs', 'IMV'])))
nlp.add_pipe('sections')
nlp.add_pipe('pollution')
```

```python
text = (
    "Le patient est admis pour des douleurs dans le bras droit, mais n'a pas de problème de locomotion. Test(et oui) "
    "Historique d'AVC dans la famille. pourrait être un cas de rhume.\n"
    "NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNBWbWbNbNbNBWbNbWbNbWBNbNbWbNbNBNbWbWbNbWBNbNbWbNBNbWbWbNb\n"
    "IMV--deshabillé\n"
    "Pourrait être un cas de rhume.\n"
    "Motif :\n"
    "-problème de locomotions==+test\n"
    "Douleurs dans le bras droit."
)
```

```python
doc = nlp(text)
```

```python
doc.ents
```

```python
doc[19]
```

```python
doc._.sections
```

```python
doc._.clean_
```

```python
doc[17]._.ascii_
```

```python
doc._.clean_
```

On peut tester l'extraction d'entité dans le texte nettoyé :

```python
doc_clean = nlp(doc._.clean_)
```

```python
ent = doc_clean[64:68]
ent
```

Les deux textes ne sont plus alignés :

```python
doc.text[ent.start_char:ent.end_char]
```

Mais la méthode `char_clean_span` permet de réaligner les deux représentations :

```python
doc._.char_clean_span(ent.start_char, ent.end_char)
```

```python

```
