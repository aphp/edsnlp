# Antecedents

The `eds.antecedents` pipeline uses a simple rule-based algorithm to detect spans that describe medical antecedent rather than the diagnostic of a given visit. It was designed at AP-HP's EDS.

The mere definition of an antecedent is not straightforward. Hence, this component only tags entities that are _explicitly described as antecedents_, eg preceded by a synonym of "antecedent".

This component may also use the output of the [`sections` pipeline](../misc/sections.md). In that case, the entire `antécédent` section is tagged as an antecedent.

!!! warning

    Be careful, the ``sections`` component may oversize the ``antécédents`` section. Indeed, it detects *section titles*
    and tags the entire text between a title and the next as a section. Hence, should a section title goes undetected after
    the ``antécédents`` title, some parts of the document will erroneously be tagged as an antecedent.

    To curb that possibility, using the output of the ``sections`` component is deactivated by default.

## Usage

The following snippet matches a simple terminology, and checks whether the extracted entities are antecedents or not. It is complete and can be run _as is_.

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.sentences")
# Dummy matcher
nlp.add_pipe(
    "eds.matcher",
    config=dict(terms=dict(douleur="douleur", malaise="malaises")),
)
nlp.add_pipe("eds.antecedents")

text = (
    "Le patient est admis le 23 août 2021 pour une douleur au bras. "
    "Il a des antécédents de malaises."
)

doc = nlp(text)

doc.ents
# Out: [patient, malaises]

doc.ents[0]._.antecedents
# Out: False

doc.ents[1]._.antecedents
# Out: True
```

## Configuration

The pipeline can be configured using the following parameters :

| Parameter      | Explanation                                                              | Default                           |
| -------------- | ------------------------------------------------------------------------ | --------------------------------- |
| `attr`         | SpaCy attribute to match on (eg `NORM`, `TEXT`, `LOWER`)                 | `"NORM"`                          |
| `antecedents`  | Antecedent patterns                                                      | `None` (use pre-defined patterns) |
| `termination`  | Termination patterns (for syntagma/proposition extraction)               | `None` (use pre-defined patterns) |
| `use_sections` | Whether to use pre-annotated sections (requires the `sections` pipeline) | `False`                           |
| `on_ents_only` | Whether to qualify pre-extracted entities only                           | `True`                            |
| `explain`      | Whether to keep track of the cues for each entity                        | `False`                           |

## Declared extensions

The `eds.antecedents` pipeline declares two [SpaCy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects :

1. The `antecedent` attribute is a boolean, set to `True` if the pipeline predicts that the span/token is an antecedent.
2. The `antecedent_` property is a human-readable string, computed from the `antecedent` attribute. It implements a simple getter function that outputs `CURRENT` or `ATCD`, depending on the value of `antecedent`.

## Authors and citation

The `eds.antecedent` pipeline was developed by AP-HP's Data Science team.
