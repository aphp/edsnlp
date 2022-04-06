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
```

```python
from edsnlp.pipelines.misc.dates.dates import Dates
from edsnlp.pipelines.misc.dates.models import AbsoluteDate, RelativeDate
from edsnlp.pipelines.misc.dates.factory import DEFAULT_CONFIG
```

# Dates

```python
nlp = spacy.blank('fr')
dates = nlp.add_pipe('eds.dates')
```

```python
text = "consultation demain le 03/09/2020 à 10h32"
```

```python
doc = nlp(text)
```

```python
ds = doc.spans['dates']
```

```python
for date in ds:
    print(date, date._.date.dict(exclude_none=True))
```

```python
for date in ds:
    print(date, date._.date.parse())
```

```python
doc = nlp("du 5 juin au 6 juillet")
```

```python
doc.spans['dates']
```

```python
for p in doc.spans['periods']:
    print(p._.period)
```

```python

```

```python
gd = dates(doc)
```

```python
AbsoluteDate.parse_obj(gd)
```

```python
import pendulum
```

```python
date
```

```python
pendulum.Duration(date.day)
```

```python
doc = nlp(text)
```

```python
doc.spans['dates']
```

```python
for date in doc.spans['dates']:
    print(date, repr(date._.date))
```

```python
def display_dates(text):
    doc = nlp(text)
    doc.ents = doc.spans['dates']
    return displacy.render(doc, style='ent', options=dict(colors=colors))
```

```python
from spacy.tokens import Span

if not Span.has_extension("groupdict"):
    Span.set_extension("groupdict", default=dict())
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
nlp('Le 12 janvier').spans['dates']
```

```python
import re
```

```python
re.search(r"(?P<covid>covid[-\s]?19)", "covid19").groupdict()
```

```python
date._.groupdict
```

```python
from edsnlp.pipelines.misc.dates.dates import parse_groupdict
```

```python
parse_groupdict(**date._.groupdict)
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
