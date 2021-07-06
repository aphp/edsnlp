---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: '[2.4.3] Py3'
    language: python
    name: pyspark-2.4.3
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
import pandas as pd
```

```python
import time
```

```python
# One-shot import of all declared Spacy components
import nlptools.components
```

```python
regex_config = {
    'douleurs':{
        'regex':[r'[Dd]ouleur'],
        'before_exclude':'azaza',
        'after_exclude':'azaza',
        'before_extract':'des',
        'after_extract':'(?:bras )(droit)'
    },
    'locomotion':{
        'regex':[r'thoracique gauche'],
        'before_include':'test'
    }
}
```

```python
nlp = spacy.blank('fr')
nlp.add_pipe('sentencizer')
nlp.add_pipe('sentences')
nlp.add_pipe('advanced_regex', config=dict(regex_config=regex_config,
                                           window=5))
nlp.add_pipe('sections')
nlp.add_pipe('pollution')
```

---

```python
text = (
    "Le patient est admis pour des douleurs dans le bras droit, mais n'a pas de problème de locomotion. "
    "Historique d'AVC dans la famille. pourrait être un cas de rhume.\n"
    "NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNBWbWbNbNbNBWbNbWbNbWBNbNbWbNbNBNbWbWbNbWBNbNbWbNBNbWbWbNb\n"
    "Pourrait être un cas de rhume.\n"
    "Motif :\n"
    "Douleurs dans le bras droit."
)
```

```python tags=[]
doc = nlp(text)
```

```python
span.text
```

```python
doc.ents[2].label_
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
doc._.clean_[165:181]
```

Les deux textes ne sont plus alignés :

```python
doc.text[165:181]
```

Mais la méthode `char_clean_span` permet de réaligner les deux représentations :

```python
span = doc._.char_clean_span(165, 181)
span
```

```python
doc._.sections[0]
```

```python

```
