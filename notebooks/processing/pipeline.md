---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
  kernelspec:
    display_name: '[2.4.3] Py3'
    language: python
    name: pyspark-2.4.3
---

```python
%reload_ext autoreload
%autoreload 2
```

```python
# Importation du "contexte", ie la bibliothèque sans installation
import context
```

```python
import spacy
import pandas as pd
```

```python
import time
from tqdm import tqdm
```

```python
# One-shot import of all declared Spacy components
import edsnlp.components
```

```python
import edsnlp.processing as nlprocess
```

```python
regex_config = {
    'douleurs':{
        'regex':[r'[Dd]ouleur'],
        'before_exclude':'azaza',
        'after_exclude':'azaza',
        'before_extract':'des',
        'after_extract':'(?:bras )(droit)'
    },
    'locomotion':{
        'regex':[r'locomotion'],
        'before_include':'test'
    }
}
```

```python
nlp = spacy.blank('fr')
nlp.add_pipe('sentences')
nlp.add_pipe('advanced_regex', config=dict(regex_config=regex_config,
                                           window=5))
nlp.add_pipe('sections')
nlp.add_pipe('pollution')
```

## 1. Pipeline sur un document unique

```python
text = (
    "Le patient est admis pour des douleurs dans le bras droit, mais n'a pas de problème de locomotion. "
    "Historique d'AVC dans la famille. pourrait être un cas de rhume.\n"
    "NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNBWbWbNbNbNBWbNbWbNbWBNbNbWbNbNBNbWbWbNbWBNbNbWbNBNbWbWbNb\n"
    "Pourrait être un cas de rhume.\n"
    "Motif :\n"
    "Douleurs dans le bras droit."
)
```

```python
doc = nlp(text)
```

Chaque `pipe` a rajouté des informations a l'objet `doc`


- Extraction via les RegEx:

```python
for entite in doc.ents:
    print(f"Label: {entite.label_} / Extraction: {entite.text} / Span: ({entite.start_char},{entite.end_char})")
```

- Extraction des phrases:

```python
for i, sent in enumerate(doc.sents):
    print(i, sent)
```

```python
doc.ents[1].sent
```

- Extraction des  sections

```python
for section in doc._.sections:
    print(f"Label: {section._.section_title} / Span: ({section.start_char},{section.end_char})")
```

## 2. Pipeline sur un petit nombre de documents

```python
notes = sql("select note_id, note_text, note_class_source_value from edsomop_prod_a.orbis_note limit 100000").toPandas()
notes = notes[notes.note_text.notna()]
```

```python
small_notes_subset = notes[:1000]
```

```python
small_notes_subset.head()
```

Les données d'entrée sont ici sous forme d'une DataFrame Pandas.
Chaque ligne va générer un objet `Doc` qui va être processé par l'objet `nlp`.
Pour cela, on peut utiliser la méthode `nlprocess.pipe`:

```python
help(nlprocess.pipe)
```

```python
%%time
ents = []
for doc in nlprocess.pipe(nlp,
                          big_notes_subset,
                          text_col='note_text',
                          context_cols=['note_id','note_class_source_value'],
                          progress_bar=True):
    if len(doc.ents) > 0:
        ents.extend(list(doc.ents))
```

```python
len(ents)
```

De la même manière qu'avec un document unique, on peut facilement accéder aux extractions, ainsi qu'au éléments de contexte renseignés via l'argument `context_cols`

```python
entite = ents[0]

print(f"Label: {entite.label_} / Extraction: {entite.text} / Span: ({entite.start_char},{entite.end_char})")
print(f"note_id: {entite.doc._.note_id}")
print(f"Type de note: {entite.doc._.note_class_source_value}")
```

## 3. Pipeline distribuée pour un grand nombre de documents


Si vous souhaitez processer un grand nombre de textes, il sera plus rapide de paralleliser le travail.
Cependant, il vous faut pour cela définir une fonction `pick_results` qui sera appelée sur chaque objet `Doc` en bout de pipeline.
La sortie de cette fonction doit être une liste de dictionnaires.
Voyons un exemple:

```python
big_notes_subset = notes
len(big_notes_subset)
```

```python
def pick_results(doc):
    """
    Function used well Paralellizing tasks via joblib
    This functions will store all extracted entities
    """
    return [{'note_id':e.doc._.note_id,
             'lexical_variant':e.text,
             'offset_start':e.start_char,
             'offset_end':e.end_char,
             'label':e.label_} for e in doc.ents if doc.ents]
```

Il suffit ensuite d'appeler la méthode `nlprocess.parallel_pipe`, qui accepte les mêmes arguments que `nlprocess.pipe` avec en plus:
- `chunksize` (int) : Taille des batchs créés pour la perallelisation
- `n_jobs` (int) : Nombre de jobs parallèles max.
- `pick_results` (func) : Voir plus haut

```python
help(nlprocess.parallel_pipe)
```

```python
%%time
ents = nlprocess.parallel_pipe(nlp,
                               big_notes_subset,
                               chunksize=100,
                               n_jobs=10,
                               context_cols='note_id',
                               progress_bar=False,
                               pick_results = pick_results)
```

---
