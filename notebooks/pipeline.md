---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: spacy
    language: python
    name: spacy
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
import negparhyp.baseline
```

# Baselines

```python
nlp = spacy.blank('fr')
```

```python
nlp.add_pipe('sentencizer')
nlp.add_pipe('sections')
nlp.add_pipe('negation')
nlp.add_pipe('hypothesis')
nlp.add_pipe('context')
```

```python
text = "Le patient est admis pour des douleurs dans le bras droit, mais n'a pas de problème de locomotion. " \
       "Historique d'AVC dans la famille. pourrait être un cas de rhume."
```

```python
doc = nlp(text)
```

```python
print(f'{"Token":<16}{"Polarity":<12} {"Hypothesis":<12} {"Context":<5}')
print(f'{"-----":<16}{"--------":<12} {"----------":<12} {"-------":<5}')

for token in doc:
    print(f'{token.text:<16}{token._.polarity_:<12} {token._.hypothesis_:<12} {token._.context_:<5}')
```

```python
doc._.family
```

```python
doc._.negations
```

```python
doc._.hypothesis
```
