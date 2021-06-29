---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.3
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
import pandas as pd
```

```python
import os
```

```python
import context
```

```python
from nlptools.utils.brat import BratConnector
```

```python
from nlptools import components
```

```python
import spacy
```

# Sections dataset


Réutilisation du [travail réalisé par Ivan Lerner à l'EDS](https://gitlab.eds.aphp.fr/IvanL/section_dataset).

```python
data_dir = '../../data/section_dataset/'
```

```python
brat = BratConnector(data_dir)
```

```python
texts, annotations = brat.get_brat()
```

```python
texts
```

```python
nlp = spacy.blank('fr')
```

```python
nlp.add_pipe('sections')
```

```python
df = texts.copy()
```

```python
df['doc'] = df.note_text.apply(nlp)
```

```python
def assign_id(row):
    row.doc._.note_id = row.note_id
```

```python
df.apply(assign_id, axis=1);
```

```python
df['matches'] = df.doc.apply(lambda d: [dict(
    lexical_variant=s.text,
    label=s.label_,
    start=s.start_char, 
    end=s.end_char
) for s in d._.section_titles])
```

```python
df = df[['note_text', 'note_id', 'matches']].explode('matches')
```

```python
df = df.dropna()
```

```python
df[['lexical_variant', 'label', 'start', 'end']] = df.matches.apply(pd.Series)
```

```python
df = df.drop('matches', axis=1)
```

```python
df.head(20)
```

```python

```
