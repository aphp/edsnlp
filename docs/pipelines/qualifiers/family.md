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
# Out: (douleur, osthéoporose)

doc.ents[0]._.family
# Out: False

doc.ents[1]._.family
# Out: True
```

## Configuration

The pipeline can be configured using the following parameters :

| Parameter      | Explanation                                                              | Default                           |
| -------------- | ------------------------------------------------------------------------ | --------------------------------- |
| `attr`         | spaCy attribute to match on (eg `NORM`, `TEXT`, `LOWER`)                 | `"NORM"`                          |
| `family`       | Family patterns                                                          | `None` (use pre-defined patterns) |
| `termination`  | Termination patterns (for syntagma/proposition extraction)               | `None` (use pre-defined patterns) |
| `use_sections` | Whether to use pre-annotated sections (requires the `sections` pipeline) | `False`                           |
| `on_ents_only` | Whether to qualify pre-extracted entities only                           | `True`                            |
| `explain`      | Whether to keep track of the cues for each entity                        | `False`                           |

## Declared extensions

The `eds.family` pipeline declares two [spaCy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects :

1. The `family` attribute is a boolean, set to `True` if the pipeline predicts that the span/token relates to a family member.
2. The `family_` property is a human-readable string, computed from the `family` attribute. It implements a simple getter function that outputs `PATIENT` or `FAMILY`, depending on the value of `family`.

## Authors and citation

The `eds.family` pipeline was developed by AP-HP's Data Science team.
