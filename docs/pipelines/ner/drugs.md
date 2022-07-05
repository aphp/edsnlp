# Drugs

The `eds.drugs` pipeline component detects mentions of French drugs (brand names and active ingredients) and adds them to `doc.ents`.
Each drug has an associated [ATC](https://en.wikipedia.org/wiki/Anatomical_Therapeutic_Chemical_Classification_System) code.
ATC classifies drugs into groups.

## Usage

In this example, we are looking for an oral antidiabetic medication (ATC code: A10B).

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.normalizer")
nlp.add_pipe("eds.drugs", config=dict(algorithm="exact"))

text = "Traitement habituel: Kardégic, cardensiel (bisoprolol), glucophage, lasilix"

doc = nlp(text)

drugs_detected = [(x.text, x.kb_id_) for x in doc.ents]

drugs_detected
# Out: [('Kardégic', 'B01AC06'), ('cardensiel', 'C07AB07'), ('bisoprolol', 'C07AB07'), ('glucophage', 'A10BA02'), ('lasilix', 'C03CA01')]

oral_antidiabetics_detected = list(
    filter(lambda x: (x[1].startswith("A10B")), drugs_detected)
)
oral_antidiabetics_detected
# Out: [('glucophage', 'A10BA02')]
```

Glucophage is the brand name of a medication that contains metformine, the first-line medication for the treatment of type 2 diabetes.

## Configuration

The pipeline can be configured using the following parameters :

| Parameter          | Description                                                    | Default   |
|--------------------|----------------------------------------------------------------|-----------|
| `algorithm`        | Which algorithm should we use : `exact` or `simstring`         | `"LOWER"` |
| `algorithm_config` | Config of the algorithm (`SimstringMatcher`'s for `simstring`) | `"LOWER"` |
| `attr`             | spaCy attribute to match on (eg `NORM`, `TEXT`, `LOWER`)       | `"NORM"`  |
| `ignore_excluded`  | Whether to ignore excluded tokens for matching                 | `False`   |

## Authors and citation

The `eds.drugs` pipeline was developed by the IAM team and CHU de Bordeaux's Data Science team.
