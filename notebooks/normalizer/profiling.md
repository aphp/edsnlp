---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.1
  kernelspec:
    display_name: 'Python 3.9.5 64-bit (''.venv'': venv)'
    language: python
    name: python3
---

```python
%reload_ext autoreload
%autoreload 2
```

```python
import context
```

```python
from edsnlp import components
```

```python
import spacy
```

# Date detection

```python
text = (
    "Le patient est admis pour des douleurs dans le bras droit, mais n'a pas de problème de locomotion. "
    "Historique d'AVC dans la famille. pourrait être un cas de rhume.\n"
    "NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNBWbWbNbNbNBWbNbWbNbWBNbNbWbNbNBNbWbWbNbWBNbNbWbNBNbWbWbNb\n"
    "Pourrait être un cas de rhume.\n"
    "Motif :\n"
    "Douleurs dans le bras droit.\n"
    "ANTÉCÉDENTS\n"
    "Le patient est déjà venu\n"
    "Pas d'anomalie détectée.\n\n"
) * 10
```

```python
nlp = spacy.blank('fr')
# nlp.add_pipe('lowercase')
# nlp.add_pipe('accents')
# nlp.add_pipe('pollution')
# nlp.add_pipe('normalizer', config=dict(lowercase=False, accents=False, pollution=False))
nlp.add_pipe('sentences')
nlp.add_pipe(
    "matcher",
    name="matcher",
    config=dict(
        attr='TEXT',
        regex=dict(anomalie=r"anomalie"),
    ),
)
nlp.add_pipe('negation', config=dict(attr='TEXT'))
```

```python
%%timeit
nlp(text)
```

```python
nlp = spacy.blank('fr')
nlp.add_pipe('lowercase')
# nlp.add_pipe('accents')
# nlp.add_pipe('pollution')
# nlp.add_pipe('normalizer', config=dict(lowercase=False, accents=False, pollution=False))
nlp.add_pipe('sentences')
nlp.add_pipe(
    "matcher",
    name="matcher",
    config=dict(
        attr='TEXT',
        regex=dict(anomalie=r"anomalie"),
    ),
)
nlp.add_pipe('negation', config=dict(attr='TEXT'))
```

```python
%%timeit
nlp(text)
```

```python
nlp = spacy.blank('fr')
nlp.add_pipe('lowercase')
nlp.add_pipe('accents')
# nlp.add_pipe('pollution')
# nlp.add_pipe('normalizer', config=dict(lowercase=False, accents=False, pollution=False))
nlp.add_pipe('sentences')
nlp.add_pipe(
    "matcher",
    name="matcher",
    config=dict(
        attr='TEXT',
        regex=dict(anomalie=r"anomalie"),
    ),
)
nlp.add_pipe('negation', config=dict(attr='TEXT'))
```

```python
%%timeit
nlp(text)
```

```python
nlp = spacy.blank('fr')
nlp.add_pipe('lowercase')
nlp.add_pipe('accents')
nlp.add_pipe('pollution')
# nlp.add_pipe('normalizer', config=dict(lowercase=False, accents=False, pollution=False))
nlp.add_pipe('sentences')
nlp.add_pipe(
    "matcher",
    name="matcher",
    config=dict(
        attr='TEXT',
        regex=dict(anomalie=r"anomalie"),
    ),
)
nlp.add_pipe('negation', config=dict(attr='TEXT'))
```

```python
%%timeit
nlp(text)
```

```python
nlp = spacy.blank('fr')
nlp.add_pipe('normalizer')
nlp.add_pipe('sentences')
nlp.add_pipe(
    "matcher",
    name="matcher",
    config=dict(
        attr='TEXT',
        regex=dict(anomalie=r"anomalie"),
    ),
)
nlp.add_pipe('negation', config=dict(attr='TEXT'))
```

```python
%%timeit
nlp(text)
```

```python
nlp = spacy.blank('fr')
# nlp.add_pipe('lowercase')
# nlp.add_pipe('accents')
# nlp.add_pipe('pollution')
# nlp.add_pipe('normalizer', config=dict(lowercase=False, accents=False, pollution=False))
nlp.add_pipe('normalizer')
nlp.add_pipe('sentences')
nlp.add_pipe(
    "matcher",
    name="matcher",
    config=dict(
        attr='CUSTOM_NORM',
        regex=dict(anomalie=r"anomalie"),
    ),
)
nlp.add_pipe('negation', config=dict(attr='TEXT'))
```

```python
%%timeit
nlp(text)
```

```python
nlp = spacy.blank('fr')
# nlp.add_pipe('lowercase')
# nlp.add_pipe('accents')
# nlp.add_pipe('pollution')
# nlp.add_pipe('normalizer', config=dict(lowercase=False, accents=False, pollution=False))
nlp.add_pipe('normalizer')
nlp.add_pipe('sentences')
nlp.add_pipe(
    "matcher",
    name="matcher",
    config=dict(
        attr='CUSTOM_NORM',
        regex=dict(anomalie=r"anomalie"),
    ),
)
nlp.add_pipe('negation', config=dict(attr='CUSTOM_NORM'))
```

```python
%%timeit
nlp(text)
```

```python
nlp = spacy.blank('fr')
# nlp.add_pipe('lowercase')
# nlp.add_pipe('accents')
# nlp.add_pipe('pollution')
# nlp.add_pipe('normalizer', config=dict(lowercase=False, accents=False, pollution=False))
nlp.add_pipe('normalizer')
nlp.add_pipe('sentences')
nlp.add_pipe(
    "matcher",
    name="matcher",
    config=dict(
        attr='CUSTOM_NORM',
        regex=dict(anomalie=r"anomalie"),
    ),
)
nlp.add_pipe('negation', config=dict(attr='CUSTOM_NORM'))
```

```python
%%timeit
nlp(text)
```

```python

```
