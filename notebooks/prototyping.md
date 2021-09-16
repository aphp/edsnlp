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
    display_name: Python 3
    language: python
    name: python3
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
# Import components without declaring them
from edsnlp.pipelines.pollution import Pollution, terms as pollution_terms
```

```python
from edsnlp.pipelines.sections import Sections, terms as section_terms
```

```python
from edsnlp.pipelines.quickumls import QuickUMLSComponent
```

```python
from edsnlp.pipelines.generic import GenericMatcher
```

# Baselines

In this notebook, we avoid declaring the components to Spacy. Hence the `autoreload` function will work properly, making prototyping way easier.

```python
nlp = spacy.blank('fr')
```

```python
nlp.add_pipe('sentencizer')
```

```python
sections = Sections(nlp, section_terms.sections, fuzzy=True)
```

```python
matcher = GenericMatcher(
    nlp,
    regex=dict(famille=[r'douleuur']),
    fuzzy=True,
    filter_matches=True,
)
```

```python
matcher2 = GenericMatcher(
    nlp,
    regex=dict(test=[r'\b\w+\b']),
    fuzzy=True,
    filter_matches=True,
    on_ents_only=True
)
```

```python
pollution = Pollution(nlp, pollution_terms.pollution)
```

```python
text = (
    "Le patient est admis pour des douleuurs dans le bras droit, mais n'a pas de problème de locomotion. "
    "Historique d'AVC dans la famille. pourrait être un cas de rhume.\n"
    "NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNBWbWbNbNbNBWbNbWbNbWBNbNbWbNbNBNbWbWbNbWBNbNbWbNBNbWbWbNb\n"
    "Pourrait être un cas de rhume. \n"
    "Motif :\n"
    "Douleurs dans le bras droit."
)
```

```python
doc = nlp(text)
```

```python
doc = matcher(doc)
```

```python
doc.ents
```

```python
doc = matcher2(doc)
```

```python
doc.ents
```

```python

```

```python
from spacy.util import filter_spans
```

```python
%timeit filter_spans(doc.ents)
```

```python
%timeit matcher._filter_matches(doc.ents)
```

```python
doc.ents
```

```python
doc.ents[0].label_
```

```python
doc._.sections
```

```python
doc._.sections
```

```python
doc = pollution(doc)
```

```python
doc._.section_titles
```

```python
doc._.sections
```

```python
{s.label_: s for s in doc._.section_titles}
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

```python

```
