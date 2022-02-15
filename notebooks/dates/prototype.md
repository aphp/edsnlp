---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: "1.3"
      jupytext_version: 1.13.5
  kernelspec:
    display_name: "Python 3.9.5 64-bit ('.venv': venv)"
    language: python
    name: python3
---

```python
import context
```

```python
from typing import Dict, Callable
```

```python
import spacy
from spacy import displacy
```

```python
from edsnlp import components
```

```python
from edsnlp.pipelines.misc.dates.dates import parse_groupdict
```

# Dates

```python
text = "Le patient est atteint venu le 29 janvier 2012"
```

```python
category20 = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]
```

```python
nlp = spacy.blank('fr')
dates = nlp.add_pipe('dates')
```

```python
labels = [
    'absolute',
    'full_date',
    'no_year',
    'no_day',
    'year_only',
    'relative',
    'current',
    'false_positive',
]
```

```python
colors = {
    label: category20[i] for i, label in enumerate(labels)
}
```

```python
def display_dates(text):
    doc = nlp(text)
    doc.ents = doc.spans['dates']
    return displacy.render(doc, style='ent', options=dict(colors=colors))
```

```python
display_dates(text)
```

```python
display_dates('Le 3 janvier 2012 à 9h')
```

```python
date = nlp('Le 12 janvier').spans['dates'][0]
```

```python
date.label_
```

```python
date._.parsed_date
```

```python
date._.groupdict
```

```python
from dateparser import DateDataParser
```

```python
parser = DateDataParser(['fr'])
```

```python
parser.get_date_data('le 12/01/2020 à 15h')
```

```python
display_dates('Cette année, le 2 janvier')
```

```python
doc = nlp(text)
```

```python
date = doc.spans['dates'][0]
```

```python
date._.groupdict
```

```python
ent = doc.ents[0]
```

```python
ent
```

```python
ent._.groupdict
```

```python

```
