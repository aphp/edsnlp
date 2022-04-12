---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%reload_ext autoreload
%autoreload 2
```

```python
import spacy
from spacy import displacy
from spacy.tokens import Doc
```

# TNM mentions

```python
nlp = spacy.blank("fr")
dates = nlp.add_pipe("eds.tnm")
```

```python
text = "patient a un pTNM : pT0N2M1"
```

```python
doc = nlp(text)
```

```python
tnms = doc.spans['tnm']
```

```python
def display_tnm(doc: Doc):
    doc.ents = doc.spans['tnm']
    return displacy.render(doc, style='ent')
```

```python
display_tnm(doc)
```

```python
for tnm in tnms:
    print(f"{str(tnm):<25}{repr(tnm._.value)}")
```

```python

```
