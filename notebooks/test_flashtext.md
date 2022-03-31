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
from edsnlp.utils.flashtext import KeywordProcessor
import pandas as pd
import spacy



```

```python
data = pd.read_table('../data/drug.target.interaction.tsv')
data = data["DRUG_NAME"]
data = data.drop_duplicates()
data = data.reset_index()
data = data["DRUG_NAME"]
```

```python
keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_list(list(data) + ["lo"] )
    
keywords_found = keyword_processor.extract_keywords('I love levobupivacaine, (S)-nicardipine and aminopterin', max_cost=2, span_info=True)
keywords_found
```

```python
nlp = spacy.blank("fr")
terms = dict(
medicament=['test1','test2']
)
nlp.add_pipe("flashtext.matcher", config=dict(terms=terms, max_cost=1))
```

```python
story = "je , suis, un tast1 et un test2"

nlp_matcher_result = list(nlp(story).ents)
```

```python
nlp_matcher_result
```

```python
text = """COMPTE RENDU D'HOSPITALISATION du 11/07/2018 au 12/07/2018
MOTIF D'HOSPITALISATION
Monsieur Dupont Jean Michel, de sexe masculin, âgée de 39 ans,
née le 23/11/1978, a été hospitalisé du 11/08/2019 au 17/08/2019
pour attaque d'asthme.

ANTÉCÉDENTS
Antécédents médicaux :
Premier épisode d'asthme en mai 2018."""

nlp = spacy.blank("fr")

# Extraction d'entités nommées
nlp.add_pipe(
    "flashtext.matcher",
    #"eds.matcher",
    config=dict(
        terms=dict(
            respiratoire=[
                "asthmatique",
                "asthme",
                "toux",
            ]
        )
    ),
)


nlp.add_pipe("eds.normalizer")
nlp.add_pipe("eds.sections")
nlp.add_pipe("eds.reason", config=dict(use_sections=True))

doc = nlp(text)

```

```python
reason = doc.spans["reasons"][0]
reason
```

```python
reason._.is_reason
```

```python
entities = reason._.ents_reason  # 

for e in entities:
    print(
        "Entity:",
        e.text,
        "-- Label:",
        e.label_,
        "-- is_reason:",
        e._.is_reason,
    )
```

```python
for e in doc.ents:
    print(e.start, e, e._.is_reason)
```
