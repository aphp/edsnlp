# Family

The `family` pipeline uses a simple rule-based algorithm to detect spans that describe a family member (or family history) of the patient rather than the patient themself. It was designed at AP-HP's EDS.

## Declared extensions

The `family` pipeline declares two [Spacy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects :

1. The `family` attribute is a boolean, set to `True` if the pipeline predicts that the span/token relates to a family member.
2. The `family_` property is a human-readable string, computed from the `family` attribute. It implements a simple getter function that outputs `PATIENT` or `FAMILY`, depending on the value of `family`.

## Usage

The following snippet matches a simple terminology, and checks the family context of the extracted entities. It is complete, and should run _as is_.

```python
import spacy
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("sentences")
nlp.add_pipe(
    "matcher",
    config=dict(terms=dict(douleur="douleur", ostheoporose="osthéoporose")),
)
nlp.add_pipe("family")

text = (
    "Le patient est admis le 23 août 2021 pour une douleur au bras. "
    "Il a des antécédents familiaux d'osthéoporose"
)

doc = nlp(text)

doc.ents
# Out: [patient, osthéoporose]

doc.ents[0]._.family_
# Out: 'PATIENT'

doc.ents[1]._.family_
# Out: 'FAMILY'
```

## Performance

The pipeline's performance are measured on the NegParHyp corpus. This dataset was specifically developed at EDS to test the pipeline on actual medical notes, using pseudonymised notes from the EDS. See [EDS-Datasets](https://equipedatascience-pages.eds.aphp.fr/eds-datasets/datasets/negparhyp.html) for more information.

| Split | Family F1 | support |
| ----- | --------- | ------- |
| train | 71%       | 83      |
| test  | 25%       | 4       |

The low performance on family labels can be explained by the low number of testing examples (4 occurrences). The F1-scores goes up to 71% on the training dataset (for 83 occurrences).

## Authors and citation

The `family` pipeline was developed by the Data Science team at EDS.
