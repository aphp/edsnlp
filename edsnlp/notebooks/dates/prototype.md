---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
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

```python
from edsnlp.utils.colors import create_colors
```

# Dates

```python
nlp = spacy.blank('fr')
dates = nlp.add_pipe('eds.dates', config=dict(detect_periods=True))
```

```python
text = "le 5 janvier à 15h32 cette année il y a trois semaines pdt 1 mois"
```

```python
doc = nlp(text)
```

```python
ds = doc.spans['dates']
```

```python
colors = create_colors(['absolute', 'relative', 'duration'])

def display_dates(doc: Doc):
    doc.ents = doc.spans['dates']
    return displacy.render(doc, style='ent', options=dict(colors=colors))
```

```python
display_dates(doc)
```

```python
for date in ds:
    print(f"{str(date):<25}{repr(date._.date)}")
```

```python
for date in ds:
    print(f"{str(date):<25}{date._.date.dict(exclude_none=True)}")
```

```python
for date in ds:
    print(f"{str(date):<25}{date._.date.to_datetime()}")
```

```python
for date in ds:
    print(f"{str(date):<25}{date._.date.norm()}")
```

```python
for p in doc.spans['periods']:
    print(f"{str(p):<40}{p._.period.dict()}")
```

```python

```
