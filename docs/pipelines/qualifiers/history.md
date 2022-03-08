# Medical History

The `eds.history` pipeline uses a simple rule-based algorithm to detect spans that describe medical history rather than the diagnostic of a given visit.

The mere definition of an medical history is not straightforward.
Hence, this component only tags entities that are _explicitly described as part of the medical history_,
eg preceded by a synonym of "medical history".

This component may also use the output of the [`eds.sections` pipeline](../misc/sections.md). In that case, the entire `antécédent` section is tagged as a medical history.

!!! warning

    Be careful, the `eds.sections` component may oversize the `antécédents` section. Indeed, it detects *section titles*
    and tags the entire text between a title and the next as a section. Hence, should a section title goes undetected after
    the `antécédents` title, some parts of the document will erroneously be tagged as a medical history.

    To curb that possibility, using the output of the `eds.sections` component is deactivated by default.

## Usage

The following snippet matches a simple terminology, and checks whether the extracted entities are history or not. It is complete and can be run _as is_.

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.sentences")
# Dummy matcher
nlp.add_pipe(
    "eds.matcher",
    config=dict(terms=dict(douleur="douleur", malaise="malaises")),
)
nlp.add_pipe("eds.history")

text = (
    "Le patient est admis le 23 août 2021 pour une douleur au bras. "
    "Il a des antécédents de malaises."
)

doc = nlp(text)

doc.ents
# Out: [douleur, malaises]

doc.ents[0]._.history
# Out: False

doc.ents[1]._.history
# Out: True
```

## Configuration

The pipeline can be configured using the following parameters :

| Parameter      | Explanation                                                              | Default                           |
| -------------- | ------------------------------------------------------------------------ | --------------------------------- |
| `attr`         | spaCy attribute to match on (eg `NORM`, `TEXT`, `LOWER`)                 | `"NORM"`                          |
| `history`      | History patterns                                                         | `None` (use pre-defined patterns) |
| `termination`  | Termination patterns (for syntagma/proposition extraction)               | `None` (use pre-defined patterns) |
| `use_sections` | Whether to use pre-annotated sections (requires the `sections` pipeline) | `False`                           |
| `on_ents_only` | Whether to qualify pre-extracted entities only                           | `True`                            |
| `explain`      | Whether to keep track of the cues for each entity                        | `False`                           |

## Declared extensions

The `eds.history` pipeline declares two [spaCy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects :

1. The `history` attribute is a boolean, set to `True` if the pipeline predicts that the span/token is a medical history.
2. The `history_` property is a human-readable string, computed from the `history` attribute. It implements a simple getter function that outputs `CURRENT` or `ATCD`, depending on the value of `history`.

## Authors and citation

The `eds.history` pipeline was developed by AP-HP's Data Science team.
