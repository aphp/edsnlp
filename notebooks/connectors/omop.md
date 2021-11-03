---
jupyter:
  jupytext:
    formats: md,ipynb
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: 'Python 3.9.5 64-bit (''.env'': venv)'
    name: python3
---

```python
import context
```

```python
import spacy
```

```python
from edsnlp.connectors.omop import OmopConnector
```

```python
from edsnlp import components
```

# Date detection

```python
text = (
    "Le patient est arrivé le 23 août (23/08/2021). "
    "Il dit avoir eu mal au ventre hier. "
    "L'année dernière, on lui avait prescrit du doliprane."
)
```

```python
nlp = spacy.blank('fr')
```

```python
nlp.add_pipe('normalizer')
nlp.add_pipe('matcher', config=dict(regex=dict(word=r"(\w+)")))
```

```python
doc = nlp(text)
```

```python
doc._.note_id = 0
```

```python
docs = []

for i in range(10):
    doc = nlp(f"Doc{i:02}" + text)
    doc._.note_id = i
    docs.append(doc)
```

```python
connector = OmopConnector(nlp)
```

```python
note, note_nlp = connector.docs2omop(docs)
```

```python
note
```

```python
new_docs = connector.omop2docs(note, note_nlp)
```

```python
new_docs[0].text == docs[0].text
```

```python
len(docs[0].ents) == len(new_docs[0].ents)
```

```python
for e, o in zip(new_docs[0].ents, docs[0].ents):
    assert e.text == o.text
```

```python

```
