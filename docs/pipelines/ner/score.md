# Score

The `eds.score` pipeline allows easy extraction of typical scores (Charlson, SOFA...) that can be found in clinical documents.
The pipeline works by

- Extracting the score's name via the provided regular expressions
- Extracting the score's _raw_ value via another set of RegEx
- Normalising the score's value via a normalising function

## Charlson Comorbidity Index

Implementing the `eds.score` pipeline, the `charlson` pipeline will extract the [Charlson Comorbidity Index](https://www.mdcalc.com/charlson-comorbidity-index-cci):

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.sentences")
nlp.add_pipe("eds.normalizer")
nlp.add_pipe("eds.charlson")

text = "Charlson à l'admission: 7.\n" "Charlson: \n" "OMS: \n"

doc = nlp(text)
doc.ents
# Out: (Charlson à l'admission: 7,)
```

We can see that only one occurrence was extracted. The second mention of Charlson in the text
doesn't contain any numerical value, so it isn't extracted.

Each extraction exposes 2 extensions:

```python
ent = doc.ents[0]

ent._.score_name
# Out: 'charlson'

ent._.score_value
# Out: 7
```

## SOFA score

The `SOFA` pipe allows to extract [Sequential Organ Failure Assessment (SOFA) scores](https://www.mdcalc.com/calc/691/sequential-organ-failure-assessment-sofa-score), used to track a person's status during the stay in an intensive care unit to determine the extent of a person's organ function or rate failure.

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.sentences")
nlp.add_pipe("eds.normalizer")
nlp.add_pipe("eds.sofa")

text = "SOFA (à 24H) : 12.\n" "OMS: \n"

doc = nlp(text)
doc.ents
# Out: (SOFA (à 24H) : 12,)
```

Each extraction exposes 3 extensions:

```python
ent = doc.ents[0]

ent._.score_name
# Out: 'sofa'

ent._.score_value
# Out: 12

ent._.score_method
# Out: '24H'
```

Score method can here be "24H", "Maximum", "A l'admission" or "Non précisée"

## TNM score

The `eds.tnm` pipe allows to extract TNM scores.

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.sentences")
nlp.add_pipe("eds.tnm")

text = "TNM: pTx N1 M1"

doc = nlp(text)
doc.ents
# Out: (pTx N1 M1,)

ent = doc.ents[0]
ent._.value.dict()
# {'modifier': 'p',
#  'tumour': None,
#  'tumour_specification': 'x',
#  'node': '1',
#  'node_specification': None,
#  'metastasis': '1',
#  'resection_completeness': None,
#  'version': None,
#  'version_year': None}
```

The TNM score is based on the developement of S. Priou, B. Rance and E. Kempf [@kempf:hal-03519085].

## Implementing your own score

Using the `eds.score` pipeline, you only have to change its configuration in order to implement a _simple_ score extraction algorithm. As an example, let us see the configuration used for the `eds.charlson` pipe
The configuration consists of 4 items:

- `score_name`: The name of the score
- `regex`: A list of regular expression to detect the score's mention
- `value_extract`: A regular expression to extract the score's value in the context of the score's mention
- `score_normalization`: A function name used to normalise the score's _raw_ value

!!! note

    spaCy doesn't allow to pass functions in the configuration of a pipeline.
    To circumvent this issue, functions need to be registered, which simply consists in
    decorating those functions

The registration is done as follows:

```python
@spacy.registry.misc("score_normalization.charlson")
def my_normalization_score(raw_score: str):
    # Implement some filtering here
    # Return None if you want the score to be discarded
    return normalized_score
```

The values used for the `eds.charlson` pipe are the following:

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
    score_name="charlson",
    regex=[r"charlson"],
    value_extract=r"charlson.*[\n\W]*(\d+)",
    score_normalization="score_normalization.charlson",
)
```

\bibliography
