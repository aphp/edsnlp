# Basic architecture of a component

Most pipelines provided by EDS-NLP aim to qualify pre-extracted entities. To wit, the basic usage of the library is :

1. Implement a normalizer (see [`normalizer`](../user-guide/normalizer.md))
2. Add an entity recognition component (eg the simple but powerful [`matcher` pipeline](../user-guide/matcher.md))
3. Add zero or more entity qualification components, such as [`negation`](../user-guide/negation.md), [`family`](../user-guide/family.md) or [`hypothesis`](../user-guide/hypothesis.md). These qualifiers typically help detect false-positives.

## Scope

Since the basic usage of EDS-NLP components is to qualify entities, most pipelines can function in two modes :

1. Annotation of the extracted entities (this is the default). To increase throughput, only pre-extracted entities (found in `doc.ents`) are processed.
2. Full-text, token-wise annotation. This mode is activated with by setting the `on_ents_only` parameter to `False`.

The possibility to do full-text annotation implies that one could use the pipelines the other way around, eg detecting all negations once and for all in an ETL phase, and reusing the results consequently. However, this is not the intended use of the library, which aims to help researchers downstream as a standalone application.

## Declared extensions

Most pipelines declare [Spacy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on the `Doc`, `Span` and/or `Token` objects. These extensions collect the output of the different pipelines.

Typically, extensions come in pairs :

1. One is computed during processing. It typically contains a boolean (eg `negation` with the `negated` extension).
2. The other comes in the form of a property computed from the latter, to provide a human-readable format. We follow Spacy naming convention and end these extensions with a trailing `_`.
