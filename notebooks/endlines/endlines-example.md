---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: "1.3"
      jupytext_version: 1.13.0
  kernelspec:
    display_name: "Python 3.7.1 64-bit ('env_debug': conda)"
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import spacy
```

```python
from edsnlp.pipelines.endlines.endlinesmodel import EndLinesModel
```

```python
import pandas as pd
```

```python
from spacy import displacy
```

# Train

```python
nlp = spacy.blank("fr")
```

```python
text =  r"""Le patient est arrivé hier soir.
Il est accompagné par son fils

ANTECEDENTS
Il a fait une TS en 2010;
Fumeur, il est arreté il a 5 mois
Chirurgie de coeur en 2011
CONCLUSION
Il doit prendre
le medicament indiqué 3 fois par jour. Revoir médecin
dans 1 mois.
DIAGNOSTIC :

Antecedents Familiaux:
- 1. Père avec diabete

"""
```

```python
doc = nlp(text)
```

```python
text2 = """J'aime le \nfromage...\n"""
doc2 = nlp(text2)
```

```python
text3 = '\nIntervention(s) - acte(s) réalisé(s) :\nParathyroïdectomie élective le [DATE]'
doc3 = nlp(text3)
```

```python
corpus = [doc,doc2, doc3]
```

```python
endlines = EndLinesModel(nlp = nlp)
```

```python
df = endlines.fit_and_predict(corpus)
df.head()
```

```python
pd.set_option("max_columns",None)
```

```python
# Save model
PATH= "/path_to_model"
endlines.save()
```

# Predict

```python
df2 = pd.DataFrame({"A1":[12646014,4191891561709484510 , 1668228190683662995],
                   "A2":[12646065887601541794,4191891561709484510 , 1668228190683662995],
                   "A3": ["UPPER","DIGIT","sdf"],
                   "A4": ["DIGIT","ENUMERATION","STRONG_PUNCT"],
                   "B1": [.5,.7,10.2],
                   "B2": [.0,.2,-10.2],
                  "BLANK_LINE":[False,True,False]})
df2 = endlines.predict(df2)
df2
```

# Set spans in training data (for viz)

```python
set_spans = endlines.set_spans
```

```python
set_spans(corpus, df)
```

```python
df.loc[df.DOC_ID==1]
```

```python
doc_exemple = corpus[1]
```

```python
doc_exemple.spans
```

```python
doc_exemple.ents = tuple(doc_exemple.spans['new_lines'])
```

```python
displacy.render(doc_exemple, style="ent", options={"colors":{"end_line":"green","space":"red"}})
```

# Pipe spacy (inference)

```python

```

```python
nlp = spacy.blank("fr")
```

```python
nlp.add_pipe("endlines", config=dict(model_path = PATH))
```

```python
docs2 = list(nlp.pipe([text,text2,text3]))
```

```python
doc_exemple = docs2[1]
```

```python
doc_exemple
```

```python
from edsnlp.utils.filter import filter_spans
spaces = tuple(s for s in doc_exemple.spans['new_lines'] if s.label_=="space")
ents = doc_exemple.ents + spaces
ents_f = filter_spans(ents)
doc_exemple.ents = ents_f
```

```python
displacy.render(doc_exemple, style="ent", options={"colors":{"space":"red"}})
```

```python

```
