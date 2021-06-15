---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
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
import pandas as pd
```

```python
# One-shot import of all declared Spacy components
from nlptools.utils.brat import BratConnector
```

# BRAT connector

```python
brat = BratConnector('../../data/section_dataset/')
```

```python
texts = brat.read_texts()
```

```python
texts.head()
```

```python
brat.read_brat_annotation('BMI_4406356.txt.txt')
```

```python
texts, annotations = brat.get_brat()
```

```python
annotations
```

```python
nlp = spacy.blank('fr')
```

```python
docs = brat.brat2docs(nlp)
```

```python
doc = docs[0]
```

```python
doc.ents
```

```python
doc.ents[0]
```

```python
annotations.head()
```

```python
brat = BratConnector('test')
```

```python
brat.docs2brat(docs)
```

```python

```
