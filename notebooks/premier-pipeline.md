---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: 'Python 3.9.5 64-bit (''.env'': venv)'
    name: python3
---

<!-- #region slideshow={"slide_type": "slide"} -->

# EDS-NLP – Présentation

<!-- #endregion -->

## Texte d'exemple

```python
with open('example.txt', 'r') as f:
    text = f.read()
```

```python
print(text)
```

## Définition d'un pipeline Spacy

```python slideshow={"slide_type": "slide"}
# Importation de Spacy
import spacy
```

```python
# Chargement des composants EDS-NLP
import edsnlp.components
```

```python
# Création de l'instance Spacy
nlp = spacy.blank('fr')

# Normalisation des accents, de la casse et autres caractères spéciaux
nlp.add_pipe('normalizer')
# Détection des fins de phrases
nlp.add_pipe('sentences')

# Extraction d'entités nommées
nlp.add_pipe(
    'matcher',
    config=dict(
        terms=dict(respiratoire=[
            'difficultes respiratoires',
            'asthmatique',
            'toux',
        ]),
        regex=dict(
            covid=r'(?i)(?:infection\sau\s)?(covid[\s\-]?19|corona[\s\-]?virus)',
            traitement=r'(?i)traitements?|medicaments?'),
        attr='NORM',
    ),
)

# Qualification des entités
nlp.add_pipe('negation')
nlp.add_pipe('hypothesis')
nlp.add_pipe('family')
nlp.add_pipe('rspeech')
```

## Application du pipeline

```python
doc = nlp(text)
```

```python
doc
```

---

Les traitements effectués par EDS-NLP (et Spacy en général) sont non-destructifs :

```python
# Non-destruction
doc.text == text
```

Pour des tâches comme la normalisation, EDS-NLP ajoute des attributs aux tokens, sans perte d'information :

```python
# Normalisation
print(f"{'texte':<15}", 'normalisation')
print(f"{'-----':<15}", '-------------')
for token in doc[3:15]:
    print(f"{token.text:<15}", f"{token.norm_}")
```

Le pipeline que nous avons appliqué a extrait des entités avec le `matcher`.


EDS-NLP étant fondée sur Spacy, on peut utiliser tous les outils proposés autour de cette bibliothèque :

```python
from spacy import displacy
```

```python
displacy.render(
    doc,
    style='ent',
    options={'colors': dict(respiratoire='green', covid='orange')},
)
```

Focalisons-nous sur la première entité :

```python
entity = doc.ents[0]
```

```python
entity
```

```python
entity._.negated
```

Le pipeline n'a pas détecté de négation pour cette entité.


Présentons les documents sous un format proche d'OMOP :

```python
import pandas as pd
```

```python
df = pd.DataFrame.from_records([
    dict(
        label=ent.label_,
        start_char=ent.start_char,
        end_char=ent.end_char,
        lexical_variant=ent.text,
        negation=ent._.negated,
        family=ent._.family,
        hypothesis=ent._.hypothesis,
        rspeech=ent._.reported_speech,
    )
    for ent in doc.ents
])
```

```python
df
```
