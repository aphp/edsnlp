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
import nlptools.rules
```

# Baselines

```python
nlp = spacy.blank('fr')
```

```python
nlp.add_pipe('sentencizer')
nlp.add_pipe('sections')
nlp.add_pipe('pollution')
```

```python
text = (
    "Le patient est admis pour des douleurs dans le bras droit, mais n'a pas de problème de locomotion. "
    "Historique d'AVC dans la famille. pourrait être un cas de rhume.\n"
    "NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNBWbWbNbNbNBWbNbWbNbWBNbNbWbNbNBNbWbWbNbWBNbNbWbNBNbWbWbNb\n"
    "Pourrait être un cas de rhume.\n"
)
```

```python
doc = nlp(text)
```

```python
doc._.clean_
```

```python
doc._.clean_[165:181]
```

```python
doc.text[165:181]
```

```python
doc._.char_clean_span(165, 181)
```

```python

```
