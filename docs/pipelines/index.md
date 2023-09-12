# Pipelines overview

EDS-NLP provides easy-to-use pipeline components.

=== "Core"

    | Pipeline                 | Description                                     |
    | ------------------------ | ----------------------------------------------- |
    | `eds.normalizer`         | Non-destructive input text normalisation        |
    | `eds.sentences`          | Better sentence boundary detection              |
    | `eds.matcher`            | A simple yet powerful entity extractor          |
    | `eds.terminology`        | A simple yet powerful terminology matcher       |
    | `eds.contextual-matcher` | A conditional entity extractor                  |
    | `eds.endlines`           | An unsupervised model to classify each end line |

=== "Qualifiers"

    See the [Qualifier overview](/pipelines/qualifiers/overview/) for more information.

    --8<-- "docs/pipelines/qualifiers/overview.md:components"

=== "Miscellaneous"

    --8<-- "docs/pipelines/misc/overview.md:components"

=== "NER"

    See the [NER overview](/pipelines/ner/overview/) for more information.

    --8<-- "docs/pipelines/ner/overview.md:components"

=== "Trainable"

    | Pipeline             | Description                                                          |
    | -------------------- | -------------------------------------------------------------------- |
    | `eds.nested-ner`     | A trainable component for nested (and classic) NER                   |
    | `eds.span-qualifier` | A trainable component for multi-class multi-label span qualification |

You can add them to your pipeline by simply calling `add_pipe`, for instance:

```python
import spacy

nlp = spacy.blank("eds")
nlp.add_pipe("eds.normalizer")
nlp.add_pipe("eds.sentences")
nlp.add_pipe("eds.tnm")
```
