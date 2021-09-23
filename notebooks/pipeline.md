---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
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
```

```python
# One-shot import of all declared Spacy components
import edsnlp.components
```

```python
a = spacy.registry.get('factories','charlson')
```

```python
a()
```

# Baselines

```python
text = (
    "Le patient est admis pour des douleurs dans le bras droit. mais n'a pas de problème de locomotion. \n"
    "Historique d'AVC dans la famille\n"
    "mais ne semble pas en être un\n"
    "Charlson 7.\n"
    "Pourrait être un cas de rhume.\n"
    "Motif :\n"
    "Douleurs dans le bras droit."
)
```

```python
nlp = spacy.blank('fr')
nlp.add_pipe('sentences')
nlp.add_pipe('normalizer')
#nlp.add_pipe('charlson')
```

```python
import thinc

registered_func = spacy.registry.get("misc", "score_norm")
```

```python
@spacy.registry.misc("score_normalization.charlson")
def score_normalization(extracted_score):
    """
    Charlson score normalization.
    If available, returns the integer value of the Charlson score.
    """
    score_range = list(range(0, 30))
    if (extracted_score is not None) and (int(extracted_score) in score_range):
        return int(extracted_score)

charlson_config = dict(
    score_name = 'charlson',
    regex = [r'charlson'],
    after_extract = r"(\d+)",
    score_normalization = "score_normalization.charlson"
)

nlp = spacy.blank('fr')
nlp.add_pipe('sentences')
nlp.add_pipe('normalizer')
nlp.add_pipe('score', config = charlson_config)
```

```python
# nlp.add_pipe('sentencizer')
nlp.add_pipe('sentences')
nlp.add_pipe('normalizer')
nlp.add_pipe('matcher', config=dict(terms=dict(douleurs=['probleme de locomotion', 'douleurs']), attr='NORM'))
nlp.add_pipe('sections')
nlp.add_pipe('pollution')
```

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

```python
doc.ents[0]._.after_snippet
```

```python
doc._.sections
```

```python
doc._.clean_
```

```python
doc[17]._.ascii_
```

```python
doc._.clean_
```

On peut tester l'extraction d'entité dans le texte nettoyé :

```python
doc._.clean_[165:181]
```

Les deux textes ne sont plus alignés :

```python
doc.text[165:181]
```

Mais la méthode `char_clean_span` permet de réaligner les deux représentations :

```python
span = doc._.char_clean_span(165, 181)
span
```

```python
doc._.sections[0]
```

```python

```
