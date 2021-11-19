# Score

The `score` pipeline allows easy extraction of typical scores (Charlson, SOFA...) that can be found in clinical documents.
The pipeline works by
- Extracting the score's name via the provided regular expressions
- Extracting the score's *raw* value via another set of RegEx
- Normalizing the score's value via a normalizing function

## An example of implemented score: The Charlson Comorbidity Index

Implementing the `score` pipeline, the `charlson` pipeline will extract the [Charlson Comorbidity Index](https://www.mdcalc.com/charlson-comorbidity-index-cci):

```python
import spacy
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("sentences")
nlp.add_pipe("normalizer")
nlp.add_pipe("charlson")

text = "Charlson à l'admission: 7.\n" "Charlson: \n" "OMS: \n"

doc = nlp(text)
doc.ents
# Out: (Charlson,)
```

We can see that only one occurrence was extracted.

```python
ent = doc.ents[0]
ent.start, ent.end
# Out: 0, 1
```

The second mention of Charlson in the text doesn't contain any numerical value, so it isn't extracted.

Each extraction exposes 2 extensions:

```python
ent._.score_name
# Out: 'charlson'

ent._.score_value
# Out: 7
```

## Another example: The SOFA score

The `SOFA` pipe allows how to extract SOFA scores.

```python
import spacy
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("sentences")
nlp.add_pipe("normalizer")
nlp.add_pipe("SOFA")

text = "SOFA (à 24H) : 12.\n" "OMS: \n"

doc = nlp(text)
doc.ents
# Out: (SOFA,)
```

Each extraction exposes 3 extensions:

```python
ent._.score_name
# Out: 'SOFA'

ent._.score_value
# Out: 12

ent._.score_method
# Out: "24H"
```

Score method can here be "24H", "Maximum", "A l'admission" or "Non précisée"

## Implementing your own score

Using the `score` pipeline, you only have to change its configuration in order to implement a *simple* score extraction algorithm. As an example, let us see the configuration used for the `charlson` pipe
The configuration consists of 4 items:
- `score_name`: The name of the score
- `regex`: A list of regular expression to detect the score's mention
- `after_extract`: A regular expression to extract the score's value after the score's mention
- `score_normalization`: A function name used to normalize the score's *raw* value

```{eval-rst}
.. note::
    SpaCy doesn't allow to pass functions in the configuration of a pipeline.
    To circumvent this issue, functions need to be registered, which simply consists in
    decorating those functions
```

The registration is done as follows:

```python
@spacy.registry.misc("score_normalization.charlson")
def my_normalization_score(raw_score: str):
    # Implement some filtering here
    # Return None if you want the score to be discarded
    return normalized_score
```

The values used for the `charlson` pipe are the following:

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
    after_extract=r"charlson.*[\n\W]*(\d+)",
    score_normalization="score_normalization.charlson",
)
```
