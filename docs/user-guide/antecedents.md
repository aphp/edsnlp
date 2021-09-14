# Antecedents

The `antecedents` pipeline uses a simple rule-based algorithm to detect spans that describe medical antecedent rather than the diagnostic of a given visit. It was designed at AP-HP's EDS.

The mere definition of an antecedent is not straightforward. Hence, this component only tags entities that are _explicitly described as antecedents_, preceded by a synonym of "antecedent".

This component may also use the output of the [`sections` pipeline](sections.md). In that case, the entire `antécédent` section is tagged as an antecedent.

```{eval-rst}
.. warning::

    Be careful, the ``sections`` component may oversize the ``antécédents`` section. Indeed, it detects *section titles*
    and tags the entire text between a title and the next as a section. Hence, should a section title goes undetected after
    the ``antécédents`` title, some parts of the document will erroneously be tagged as an antecedent.

    To curb that possibility, using the output of the ``sections`` component is deactivated by default.
```

## Scope

The `antecedents` pipeline can functions in two modes :

1. Annotation of the extracted entities (this is the default). To increase throughput, only pre-extracted entities (found in `doc.ents`) are processed.
2. Full-text, token-wise annotation. This mode is activated with by setting the `on_ents_only` parameter to `False`.

Since the natural way to use EDS-NLP is to extract entities and then check whether they are related to the patient themself, the second mode is generally unused.

## Declared extensions

The `antecedents` pipeline declares two [Spacy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects :

1. The `antecedent` attribute is a boolean, set to `True` if the pipeline predicts that the span/token is an antecedent.
2. The `antecedent_` property is a human-readable string, computed from the `antecedent` attribute. It implements a simple getter function that outputs `CURRENT` or `ATCD`, depending on the value of `antecedent`.

## Usage

The following snippet matches a simple terminology, and checks whether the extracted entities are antecedents or not. It is complete, and should run _as is_.

```python
import spacy
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("sentences")
nlp.add_pipe(
    "matcher",
    config=dict(terms=dict(douleur="douleur", malaise="malaises")),
)
nlp.add_pipe("antecedents")

text = (
    "Le patient est admis le 23 août 2021 pour une douleur au bras. "
    "Il a des antécédents de malaises."
)

doc = nlp(text)

doc.ents
# Out: [patient, malaises]

doc.ents[0]._.antecedent_
# Out: 'CURRENT'

doc.ents[1]._.antecedent_
# Out: 'ATCD'
```

## Performance

The pipeline's performance is still being evaluated.

## Authors and citation

The `antecedent` pipeline was developed by the Data Science team at EDS.
