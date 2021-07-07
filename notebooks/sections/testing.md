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
from edsnlp.utils.brat import BratConnector
```

```python
from edsnlp import components
```

```python
import spacy
```

# Sections dataset


We are using [Ivan Lerner's work at EDS](https://gitlab.eds.aphp.fr/IvanL/section_dataset). Make sure you clone the repo.

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
nlp.add_pipe('normaliser')
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
df = df.rename(columns={'start': 'offset_begin', 'end': 'offset_end', 'label': 'label_value'})
```

```python
df['label_name'] = df.label_value
```

```python
df['modifier_type'] = ''
df['modifier_result'] = ''
```

```python
from ipywidgets import Output, Button, VBox, Layout, Text, HTML
from IPython.display import display
from labeltool.labelling import GlobalLabels, Labels, Labelling

out = Output()
```

```python
labels = Labels()

for label in df.label_value.unique():
    labels.add(name = label, 
               color = 'green',
               selection_type = 'button')
```

```python
labeller = Labelling(
    df,
    save_path='testing.pickle',
    labels_dict=labels.dict,
    from_save=True,
    out=out, 
    display=display,
)
```

```python
labeller.run()
out
```

```python

```
