# External Information & Context qualifiers

This tutorial shows the use of two pipes to qualify spans or entities by using the `ContextualQualifier` and the `ExternalInformationQualifier`

### Import dependencies
```python
import datetime

import pandas as pd

import edsnlp
from edsnlp.pipes.qualifiers.contextual.contextual import (
    ClassPatternsContext,
    ContextualQualifier,
)
from edsnlp.pipes.qualifiers.external_information.external_information import (
    ExternalInformation,
    ExternalInformationQualifier,
)
from edsnlp.utils.collections import get_deep_attr
```

### Data
Lets start creating a toy example
```python
# Create context dates
# The elements under this attribute should be a list of dicts with keys value and class
context_dates = [
    {
        "value": datetime.datetime(2024, 2, 15),
        "class": "Magnetic resonance imaging (procedure)",
    },
    {"value": datetime.datetime(2024, 2, 17), "class": "Biopsy (procedure)"},
    {"value": datetime.datetime(2024, 2, 17), "class": "Colonoscopy (procedure)"},
]

# Texy
text = """
RCP du 18/12/2024 : DUPONT Jean

Homme de 68 ans adressé en consultation d’oncologie pour prise en charge d’une tumeur du colon.
Antécédents : HTA, diabète de type 2, dyslipidémie, tabagisme actif (30 PA), alcoolisme chronique (60 g/jour).

Examen clinique : patient en bon état général, poids 80 kg, taille 1m75.


HISTOIRE DE LA MALADIE :
Lors du PET-CT (14/02/2024), des dépôts pathologiques ont été observés qui coïncidaient avec les résultats du scanner.
Le 15/02/2024, une IRM a été réalisée pour évaluer l’extension de la tumeur.
Une colonoscopie a été réalisée le 17/02/2024 avec une biopsie d'adénopathie sous-carinale.
Une deuxième a été biopsié le 18/02/2024. Les résultats de la biopsie ont confirmé un adénocarcinome du colon.
Il a été opéré le 20/02/2024. L’examen anatomopath ologique de la pièce opératoire a confirmé un adénocarcinome du colon stade IV avec métastases hépatiques et pulmonaires.
Trois mois après la fin du traitement de chimiothérapie (abril 2024), le patient a signalé une aggravation progressive des symptômes

CONCLUSION :  Adénocarcinome du colon stade IV avec métastases hépatiques et pulmonaires.
"""


# Create a toy dataframe
df = pd.DataFrame.from_records(
    [
        {
            "person_id": 1,
            "note_id": 1,
            "note_text": text,
            "context_dates": context_dates,
        }
    ]
)
df
```

### Define the nlp pipeline
```python
import edsnlp.pipes as eds

nlp = edsnlp.blank("eds")

nlp.add_pipe(eds.sentences())
nlp.add_pipe(eds.normalizer())
nlp.add_pipe(eds.dates())


nlp.add_pipe(
    ContextualQualifier(
        span_getter="dates",
        patterns={
            "lf1": {
                "Magnetic resonance imaging (procedure)": ClassPatternsContext(
                    **{
                        "terms": {"irm": ["IRM", "imagerie par résonance magnétique"]},
                        "regex": None,
                        "context_words": 0,
                        "context_sents": 1,
                        "attr": "TEXT",
                    }
                )
            },
            "lf2": {
                "Biopsy (procedure)": {
                    "regex": {"biopsy": ["biopsie", "biopsié"]},
                    "context_words": (10, 10),
                    "context_sents": 0,
                    "attr": "TEXT",
                }
            },
            "lf3": {
                "Surgical procedure (procedure)": {
                    "regex": {"chirurgie": ["chirurgie", "exerese", "opere"]},
                    "context_words": 0,
                    "context_sents": (2, 2),
                    "attr": "NORM",
                },
            },
        },
    )
)

nlp.add_pipe(
    ExternalInformationQualifier(
        nlp=nlp,
        span_getter="dates",
        external_information={
            "lf4": ExternalInformation(
                doc_attr="_.context_dates",
                span_attribute="_.date.to_datetime()",
                threshold=datetime.timedelta(days=0),
            )
        },
    )
)
```

### Apply the pipeline to texts
```python
doc_iterator = edsnlp.data.from_pandas(
    df, converter="omop", doc_attributes=["context_dates"]
)

docs = list(nlp.pipe(doc_iterator))
```

### Lets inspect the results
```python
doc = docs[0]
dates = doc.spans["dates"]

for date in dates:
    for attr in ["lf1", "lf2", "lf3", "lf4"]:
        value = get_deep_attr(date, "_." + attr)

        if value:
            print(date.start, date.end, date, attr, value)
```

```python
# Out : 120 125 15/02/2024 lf1 Magnetic resonance imaging (procedure)
# Out : 120 125 15/02/2024 lf4 ['Magnetic resonance imaging (procedure)']
# Out : 147 152 17/02/2024 lf2 Biopsy (procedure)
# Out : 147 152 17/02/2024 lf4 ['Biopsy (procedure)', 'Colonoscopy (procedure)']
# Out : 168 173 18/02/2024 lf2 Biopsy (procedure)
# Out : 192 197 20/02/2024 lf3 Surgical procedure (procedure)
```
