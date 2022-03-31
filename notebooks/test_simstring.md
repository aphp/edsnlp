---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: dldiy
    language: python
    name: python3
---

```python
import pandas as pd
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.feature_extractor.word_ngram import WordNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher

```

```python
data = pd.read_table('../data/drug.target.interaction.tsv')
data = data["DRUG_NAME"]
data = data.drop_duplicates()
data = data.reset_index()
data = data["DRUG_NAME"]
```

```python
db = DictDatabase(CharacterNgramFeatureExtractor())
for medoc in data :
    db.add(medoc)

searcher = Searcher(db, CosineMeasure())
text = 'I love levobupivacae and aminpterin'
results = []
for query in text.split(" ") :
    results.append(searcher.search(query, 0.7))
print(results)
```

```python

```
