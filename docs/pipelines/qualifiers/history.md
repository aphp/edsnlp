# Medical History

The `eds.history` pipeline uses a simple rule-based algorithm to detect spans that describe medical history rather than the diagnostic of a given visit.

The mere definition of an medical history is not straightforward.
Hence, this component only tags entities that are _explicitly described as part of the medical history_,
eg preceded by a synonym of "medical history".

This component may also use the output of:

- the [`eds.sections` pipeline](../misc/sections.md). In that case, the entire `antécédent` section is tagged as a medical history.

!!! warning "Sections"

    Be careful, the `eds.sections` component may oversize the `antécédents` section. Indeed, it detects *section titles*
    and tags the entire text between a title and the next as a section. Hence, should a section title goes undetected after
    the `antécédents` title, some parts of the document will erroneously be tagged as a medical history.

    To curb that possibility, using the output of the `eds.sections` component is deactivated by default.

- the [`eds.dates` pipeline](../misc/dates.md). In that case, it will take the dates into account to tag extracted entities as a medical history or not.

!!! info "Dates"

    To take the most of the `eds.dates` component, you may add the ``note_datetime`` context (cf. [Adding context][using-eds-nlps-helper-functions]). It allows the pipeline to compute the duration of absolute dates (eg le 28 août 2022/August 28, 2022). The ``birth_datetime`` context allows the pipeline to exclude the birth date from the extracted dates.

## Usage

The following snippet matches a simple terminology, and checks whether the extracted entities are history or not. It is complete and can be run _as is_.

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.sentences")
nlp.add_pipe("eds.normalizer")
nlp.add_pipe("eds.sections")
nlp.add_pipe("eds.dates")
nlp.add_pipe(
    "eds.matcher",
    config=dict(terms=dict(douleur="douleur", malaise="malaises")),
)
nlp.add_pipe(
    "eds.history",
    config=dict(
        use_sections=True,
        use_dates=True,
    ),
)

text = (
    "Le patient est admis le 23 août 2021 pour une douleur au bras. "
    "Il a des antécédents de malaises."
    "ANTÉCÉDENTS : "
    "- le patient a déjà eu des malaises. "
    "- le patient a eu une douleur à la jambe il y a 10 jours"
)

doc = nlp(text)

doc.ents
# Out: (douleur, malaises, malaises, douleur)

doc.ents[0]._.history
# Out: False

doc.ents[1]._.history
# Out: True

doc.ents[2]._.history  # (1)
# Out: True

doc.ents[3]._.history  # (2)
# Out: False
```

1. The entity is in the section `antécédent`.
2. The entity is in the section `antécédent`, however the extracted `relative_date` refers to an event that took place within 14 days.
## Configuration

The pipeline can be configured using the following parameters :

::: edsnlp.pipelines.qualifiers.history.factory.create_component
    options:
        only_parameters: true

## Declared extensions

The `eds.history` pipeline declares two [spaCy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects :

1. The `history` attribute is a boolean, set to `True` if the pipeline predicts that the span/token is a medical history.
2. The `history_` property is a human-readable string, computed from the `history` attribute. It implements a simple getter function that outputs `CURRENT` or `ATCD`, depending on the value of `history`.

## Authors and citation

The `eds.history` pipeline was developed by AP-HP's Data Science team.
