---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.4
  kernelspec:
    display_name: 'Python 3.9.5 64-bit (''.venv'': venv)'
    language: python
    name: python3
---

```python
import context
```

```python
import spacy
from spacy.matcher import Matcher
```

```python
from spacy.tokens import Span
```

```python
from edsnlp.matchers.exclusion import ExclusionMatcher
```

```python
from edsnlp import components
```

```python
Span.set_extension('normalized_variant', getter=lambda s: ''.join([t.text + t.whitespace_ for t in s if not t._.excluded]).rstrip(' '))
```

# Test normalisation

```python
text = "Le patient est atteint d'une pneumopathie à NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNB coronavirus"
```

```python
phrase = "pneumopathie à coronavirus"
```

## Clean doc method

```python
nlp = spacy.blank('fr')
nlp.add_pipe('normalizer', config=dict(lowercase=False, quotes=False, accents=False, pollution=True))
nlp.add_pipe('matcher', config=dict(terms=dict(covid=[phrase]), attr="CUSTOM_NORM"))
```

```python
%timeit doc = nlp(text)
```

## Matcher method

```python
nlp = spacy.blank('fr')
nlp.add_pipe('pollution')
```

```python
def set_ents(doc, matcher):
    doc.ents = list(matcher(doc, as_spans=True))
```

```python
doc_pattern = nlp(phrase)
```

```python
matcher = ExclusionMatcher(nlp.vocab, attr="LOWER")
```

```python
matcher.add('covid', [doc_pattern])
```

```python
%%timeit
doc = nlp(text)
set_ents(doc, matcher)
```

```python
doc = nlp(text)
set_ents(doc, matcher)
```

```python
for token in doc:
    print(token.norm_, token._.excluded)
```

```python
doc.ents[0]._.normalized_variant
```

```python

```
