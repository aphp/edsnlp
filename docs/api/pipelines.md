# Pipelines

## Matcher

```{eval-rst}
.. automodule:: edsnlp.pipelines.core.matcher.matcher
```

## Advanced matcher

```{eval-rst}
.. automodule:: edsnlp.pipelines.core.advanced.advanced
```

## Sentences

```{eval-rst}
.. automodule:: edsnlp.pipelines.core.sentences.sentences
```

## Dates

```{eval-rst}
.. automodule:: edsnlp.pipelines.misc.dates.dates
```

## Sections

```{eval-rst}
.. automodule:: edsnlp.pipelines.misc.sections.Sections
```

## Consultation Dates

```{eval-rst}
.. automodule:: edsnlp.pipelines.misc.consultation_dates.consultation_dates
```

## Normalisation

### Lowercase

```{eval-rst}
.. automodule:: edsnlp.pipelines.core.normalizer.lowercase.factory
```

### Accents

```{eval-rst}
.. automodule:: edsnlp.pipelines.core.normalizer.accents.accents
```

### Quotes

```{eval-rst}
.. automodule:: edsnlp.pipelines.core.normalizer.quotes.quotes
```

### Pollution

```{eval-rst}
.. automodule:: edsnlp.pipelines.core.normalizer.pollution.pollution
```

## End Lines

```{eval-rst}
.. automodule:: edsnlp.pipelines.core.endlines.endlines
```

```{eval-rst}
.. automodule:: edsnlp.pipelines.core.endlines.endlinesmodel
```

## Scores

### Base class

```{eval-rst}
.. automodule:: edsnlp.pipelines.ner.scores.base_score
```

### Charlson Comorbidity Index

The `charlson` pipeline implements the above `score` pipeline with the following parameters:

```python
regex = [r"charlson"]

after_extract = r"charlson.*[\n\W]*(\d+)"

score_normalization_str = "score_normalization.charlson"


@spacy.registry.misc(score_normalization_str)
def score_normalization(extracted_score: Union[str, None]):
    """
    Charlson score normalization.
    If available, returns the integer value of the Charlson score.
    """
    score_range = list(range(0, 30))
    if (extracted_score is not None) and (int(extracted_score) in score_range):
        return int(extracted_score)
```

### SOFA Score

The `SOFA` pipeline extracts the SOFA score. each extracted entity exposes three extentions:

- `score_name`: Set to `"SOFA"`
- `score_value`: The SOFA score numerical value
- `score_method`: How/When the SOFA score was obtained. Possible values are:
  - `"Maximum"`
  - `"24H"`
  - `"A l'admission"`

### Emergency Scores

Three typical emergency scores are available.
```{note}
It is designed to work **ONLY on consultation notes** (`CR-URG`), so please filter accordingly before proceeding.
```

#### **PRIORITY** Score (`"emergency.priority"`)

This pipe extracts the priority score mentionned by the IAO


#### **CCMU** Score (`"emergency.ccmu"`)

| CCMU Score | Description                                                                                                                                                               |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1          | Etat lésionnel et/ou pronostic fonctionnel jugés stables. Abstention d'acte complémentaire diagnostique ou thérapeutique à réaliser par le SMUR ou un service d'urgences. |
| 2          | Etat lésionnel et/ou pronostic fonctionnel jugés stables. Décision d'acte complémentaire diagnostique ou thérapeutique à réaliser par le SMUR ou un service d'urgences.   |
| 3          | Etat lésionnel et/ou pronostic fonctionnel jugés susceptibles de s'aggraver aux urgences ou durant l'intervention SMUR, sans mise en jeu du pronostic vital.              |
| 4          | Situation pathologique engageant le pronostic vital. Prise en charge ne comportant pas de manœuvres de réanimation immédiate.                                             |
| 5          | Situation pathologique engageant le pronostic vital. Prise en charge comportant la pratique immédiate de manœuvres de réanimation.                                        |

#### **GEMSA** Score (`"emergency.gemsa"`)

| GEMSA Score | Description                                                                                                                                                                                                                                                     |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1           | Patient décédé à l'arrivée ou avant tout geste de réanimation.                                                                                                                                                                                                  |
| 2           | Patient non convoqué, sortant après consultation ou soins (petite chirurgie, consultation médicale...).                                                                                                                                                         |
| 3           | Patient convoqué pour des soins à distance de la prise en charge initiale (surveillance de plâtre, réfection de pansement, ablation de fils, rappel de vaccination, etc.).                                                                                      |
| 4           | Patient non attendu dans un service et hospitalisé après passage au service d'accueil des urgences (SAU).                                                                                                                                                       |
| 5           | Patient attendu dans un service, ne passant au service d'accueil des urgences (SAU) que pour des raisons d'organisation (enregistrement administratif, réalisation d'un « bilan d'entrée », refus de certains services de réaliser des entrées directes, etc.). |
| 6           | Patient nécessitant une prise en charge thérapeutique immédiate importante (réanimation) ou prolongée (surveillance médicale attentive pendant au moins une heure).                                                                                             |
