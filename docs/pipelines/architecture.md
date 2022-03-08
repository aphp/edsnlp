# Basic Architecture

Most pipelines provided by EDS-NLP aim to qualify pre-extracted entities. To wit, the basic usage of the library:

1. Implement a normalizer (see [`normalizer`](./core/normalisation.md))
2. Add an entity recognition component (eg the simple but powerful [`matcher` pipeline](./core/matcher.md))
3. Add zero or more entity qualification components, such as [`negation`](./qualifiers/negation.md), [`family`](./qualifiers/family.md) or [`hypothesis`](./qualifiers/hypothesis.md). These qualifiers typically help detect false-positives.

## Scope

Since the basic usage of EDS-NLP components is to qualify entities, most pipelines can function in two modes:

1. Annotation of the extracted entities (this is the default). To increase throughput, only pre-extracted entities (found in `doc.ents`) are processed.
2. Full-text, token-wise annotation. This mode is activated by setting the `on_ents_only` parameter to `False`.

The possibility to do full-text annotation implies that one could use the pipelines the other way around, eg detecting all negations once and for all in an ETL phase, and reusing the results consequently. However, this is not the intended use of the library, which aims to help researchers downstream as a standalone application.

## Result persistence

Depending on their purpose (entity extraction, qualification, etc), EDS-NLP pipelines write their results to `Doc.ents`, `Doc.spans` or in a custom attribute.

### Extraction pipelines

Extraction pipelines (matchers, the date detector or NER pipelines, for instance) keep their results to the `Doc.ents` attribute directly.

Note that spaCy prohibits overlapping entities within the `Doc.ents` attribute. To circumvent this limitation, we [filter spans][edsnlp.utils.filter.filter_spans], and keep all discarded entities within the `discarded` key of the `Doc.spans` attribute.

Some pipelines write their output to the `Doc.spans` dictionary. We enforce the following doctrine:

- Should the pipe extract entities that are directly informative (typically the output of the [`eds.matcher`](./core/matcher.md) component), said entities are stashed in the `Doc.ents` attribute.
- On the other hand, should the entity be useful to another pipe, but less so in itself (eg the output of the [`eds.sections`](./misc/sections.md) or [`eds.dates`](./misc/dates.md) component), it will be stashed in a specific key within the `Doc.spans` attribute.

### Entity tagging

Moreover, most pipelines declare [spaCy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on the `Doc`, `Span` and/or `Token` objects.

These extensions are especially useful for qualifier pipelines, but can also be used by other pipelines to persist relevant information. For instance, the `eds.dates` pipeline:

1. Populates `#!python Doc.spans["dates"]`
2. For each detected item, keeps the normalised date in `#!python Span._.date`
