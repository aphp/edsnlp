# Pipelines

## Matcher

```{eval-rst}
.. automodule:: edsnlp.pipelines.matcher.matcher
```

## Advanced matcher

```{eval-rst}
.. automodule:: edsnlp.pipelines.advanced.advanced
```

## Sentences

```{eval-rst}
.. automodule:: edsnlp.pipelines.sentences.sentences
```

## Dates

```{eval-rst}
.. automodule:: edsnlp.pipelines.dates.dates
```

## Sections

```{eval-rst}
.. automodule:: edsnlp.pipelines.sections.sections
```

## Negation

```{eval-rst}
.. automodule:: edsnlp.pipelines.negation.negation
```

## Family

```{eval-rst}
.. automodule:: edsnlp.pipelines.family.family
```

## Hypothesis

```{eval-rst}
.. automodule:: edsnlp.pipelines.hypothesis.hypothesis
```

## Antecedents

```{eval-rst}
.. automodule:: edsnlp.pipelines.antecedents.antecedents
```

## Reported Speech

```{eval-rst}
.. automodule:: edsnlp.pipelines.rspeech.rspeech
```

### Lowercase

```{eval-rst}
.. automodule:: edsnlp.pipelines.normalizer.lowercase.factory
```

### Accents

```{eval-rst}
.. automodule:: edsnlp.pipelines.normalizer.accents.accents
```

### Quotes

```{eval-rst}
.. automodule:: edsnlp.pipelines.normalizer.quotes.quotes
```

### Pollution

```{eval-rst}
.. automodule:: edsnlp.pipelines.normalizer.pollution.pollution
```

## End Lines

```{eval-rst}
.. automodule:: edsnlp.pipelines.endlines.endlines
```

```{eval-rst}
.. automodule:: edsnlp.pipelines.endlines.endlinesmodel
```

## Scores

### Base class

```{eval-rst}
.. automodule:: edsnlp.pipelines.scores.base_score
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

```

```
