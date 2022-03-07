# Family

The `eds.family` pipeline uses a simple rule-based algorithm to detect spans that describe a family member (or family history) of the patient rather than the patient themself.

## Usage

The following snippet matches a simple terminology, and checks the family context of the extracted entities. It is complete, and can be run _as is_.

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.sentences")
# Dummy matcher
nlp.add_pipe(
    "eds.matcher",
    config=dict(terms=dict(douleur="douleur", ostheoporose="osthéoporose")),
)
nlp.add_pipe("eds.family")

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

## Configuration

The pipeline can be configured using the following parameters :

| Parameter      | Explanation                                                              | Default                           |
| -------------- | ------------------------------------------------------------------------ | --------------------------------- |
| `attr`         | SpaCy attribute to match on (eg `NORM`, `TEXT`, `LOWER`)                 | `"NORM"`                          |
| `family`       | Family patterns                                                          | `None` (use pre-defined patterns) |
| `termination`  | Termination patterns (for syntagma/proposition extraction)               | `None` (use pre-defined patterns) |
| `use_sections` | Whether to use pre-annotated sections (requires the `sections` pipeline) | `False`                           |
| `on_ents_only` | Whether to qualify pre-extracted entities only                           | `True`                            |
| `explain`      | Whether to keep track of the cues for each entity                        | `False`                           |

## Declared extensions

The `eds.amily` pipeline declares two [SpaCy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects :

1. The `family` attribute is a boolean, set to `True` if the pipeline predicts that the span/token relates to a family member.
2. The `family_` property is a human-readable string, computed from the `family` attribute. It implements a simple getter function that outputs `PATIENT` or `FAMILY`, depending on the value of `family`.

## Performance

The pipeline's performance is measured on the NegParHyp corpus. This dataset was specifically developed at EDS to test the pipeline on actual clinical notes, using pseudonymised notes from the EDS.

| Split | Family F1 | support |
| ----- | --------- | ------- |
| train | 71%       | 83      |
| test  | 25%       | 4       |

The low performance on family labels can be explained by the low number of testing examples (4 occurrences). The F1-scores goes up to 71% on the training dataset (for 83 occurrences). More extensive validation is needed to get a reliable estimation of the pipeline's generalisation capabilities.

## Authors and citation

The `eds.family` pipeline was developed by AP-HP's Data Science team.
