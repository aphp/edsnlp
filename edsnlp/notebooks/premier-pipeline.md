---
jupyter:
  jupytext:
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: "1.3"
      jupytext_version: 1.13.5
  kernelspec:
    display_name: "Python 3.9.5 64-bit ('.env': venv)"
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

## Définition d'un pipeline spaCy

```python slideshow={"slide_type": "slide"}
# Importation de spaCy
import spacy
```

```python
# Chargement des composants EDS-NLP

```

```python
# Création de l'instance spaCy
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

nlp.add_pipe('dates')

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

Les traitements effectués par EDS-NLP (et spaCy en général) sont non-destructifs :

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

Les entités détectées se retrouvent dans l'attribut `ents` :

```python
doc.ents
```

EDS-NLP étant fondée sur spaCy, on peut utiliser tous les outils proposés autour de cette bibliothèque :

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

Chaque entité a été qualifiée par les pipelines de négation, hypothèse, etc. Ces pipelines utilisent des extensions spaCy pour stocker leur résultat :

```python
entity._.negated
```

Le pipeline n'a pas détecté de négation pour cette entité.

## Application du pipleline sur une table de textes

Les textes seront le plus souvent disponibles sous la forme d'un DataFrame pandas, qu'on peut simuler ici :

```python
import pandas as pd
```

```python
note = pd.DataFrame(dict(note_text=[text] * 10))
note['note_id'] = range(len(note))
note = note[['note_id', 'note_text']]
```

```python
note
```

On peut appliquer la pipeline à l'ensemble des documents en utilisant la fonction `nlp.pipe`, qui permet d'accélérer les traitements en les appliquant en parallèle :

```python
# Ici on crée une liste qui va contenir les documents traités par spaCy
docs = list(nlp.pipe(note.note_text))
```

On veut récupérer les entités détectées et les information associées (empans, qualification, etc) :

```python
def get_entities(doc):
    """Extract a list of qualified entities from a spaCy Doc object"""
    entities = []

    for ent in doc.ents:
        entity = dict(
            start=ent.start_char,
            end=ent.end_char,
            label=ent.label_,
            lexical_variant=ent.text,
            negated=ent._.negated,
            hypothesis=ent._.hypothesis,
        )

        entities.append(entity)

    return entities
```

```python
note['entities'] = [get_entities(doc) for doc in nlp.pipe(note.note_text)]
```

```python
note
```

On peut maintenant récupérer les entités détectées au format `NOTE_NLP` (ou similaire) :

```python
# Sélection des colonnes
note_nlp = note[['note_id', 'entities']]

# "Explosion" des listes d'entités, et suppression des lignes vides (documents sans entité)
note_nlp = note_nlp.explode('entities').dropna()

# Re-création de l'index, pour des raisons internes à pandas
note_nlp = note_nlp.reset_index(drop=True)
```

```python
note_nlp
```

Il faut maintenant passer d'une colonne de dictionnaires à une table `NOTE_NLP` :

```python
note_nlp = note_nlp[['note_id']].join(pd.json_normalize(note_nlp.entities))
```

```python
note_nlp
```

On peut aggréger la qualification des entités en une unique colonne :

```python
# Création d'une colonne "discard" -> si l'entité est niée ou hypothétique, on la supprime des résultats
note_nlp['discard'] = note_nlp[['negated', 'hypothesis']].max(axis=1)
```

```python
note_nlp
```

```python

```

```python

```

```python

```

```python
%%timeit
for text in texts:
    nlp(text)
```

```python
%%timeit
for text in nlp.pipe(texts, n_process=-1):
    pass
```
