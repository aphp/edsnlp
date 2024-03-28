# Pipes overview

EDS-NLP provides easy-to-use pipeline components (aka pipes).

## Available components

<!-- --8<-- [start:components] -->

=== "Core"

    See the [Core components overview](/pipes/misc/overview/) for more information.

    --8<-- "docs/pipes/core/index.md:components"

=== "Qualifiers"

    See the [Qualifiers overview](/pipes/qualifiers/overview/) for more information.

    --8<-- "docs/pipes/qualifiers/index.md:components"

=== "Miscellaneous"

    See the [Miscellaneous components overview](/pipes/misc/overview/) for more information.

    --8<-- "docs/pipes/misc/index.md:components"

=== "NER"

    See the [NER overview](/pipes/ner/overview/) for more information.

    --8<-- "docs/pipes/ner/index.md:components"

=== "Trainable"

    See the [Trainable components overview](/pipes/trainable/overview/) for more information.

    --8<-- "docs/pipes/trainable/index.md:components"

<!-- --8<-- [end:components] -->

You can add them to your pipeline by simply calling `add_pipe`, for instance:

```python
import edsnlp, edsnlp.pipes as eds

nlp = edsnlp.blank("eds")
nlp.add_pipe(eds.normalizer())
nlp.add_pipe(eds.sentences())
nlp.add_pipe(eds.tnm())
```

## Basic architecture

Most components provided by EDS-NLP aim to qualify pre-extracted entities. To wit, the basic usage of the library:

1. Implement a normaliser (see `eds.normalizer`)
2. Add an entity recognition component (eg the simple but powerful `eds.matcher` component)
3. Add zero or more entity qualification components, such as `eds.negation`, `eds.family` or `eds.hypothesis`. These qualifiers typically help detect false-positives.

## Extraction components

Extraction components (matchers, the date detector or NER components, for instance) keep their results to the `doc.ents` and `doc.spans` attributes directly.

By default, some components do not write their output to `doc.ents`, such as the `eds.sections` matcher. This is mainly due to the fact that, since `doc.ents` cannot contain overlapping entities, we [filter spans][edsnlp.utils.filter.filter_spans] and keep the largest one by default. Since sections usually cover large spans of text, storing them in ents would remove every other overlapping entities.

## Entity tagging

Moreover, most components declare [extensions](https://spacy.io/usage/processing-components#custom-components-attributes), on the `Doc`, `Span` and/or `Token` objects.

These extensions are especially useful for qualifier components, but can also be used by other components to persist relevant information. For instance, the `eds.dates` component declares a `span._.date` extension to store a normalised version of each detected date.
