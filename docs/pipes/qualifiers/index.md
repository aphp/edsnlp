# Qualifier Overview

In EDS-NLP, we call _qualifiers_ the suite of components designed to _qualify_ a
pre-extracted entity for a linguistic modality.

## Available components

<!-- --8<-- [start:components] -->

| Pipeline              | Description                          |
|-----------------------|--------------------------------------|
| `eds.negation`        | Rule-based negation detection        |
| `eds.family`          | Rule-based family context detection  |
| `eds.hypothesis`      | Rule-based speculation detection     |
| `eds.reported_speech` | Rule-based reported speech detection |
| `eds.history`         | Rule-based medical history detection |

<!-- --8<-- [end:components] -->

## Rationale

In a typical medical NLP pipeline, a group of clinicians would define a list of synonyms for a given concept of interest (say, for example, diabetes), and look for that terminology in a corpus of documents.

Now, consider the following example:

=== "French"

    ```
    Le patient n'est pas diabétique.
    Le patient est peut-être diabétique.
    Le père du patient est diabétique.
    ```

=== "English"

    ```
    The patient is not diabetic.
    The patient could be diabetic.
    The patient's father is diabetic.
    ```

There is an obvious problem: none of these examples should lead us to include this particular patient into the cohort.

!!! warning

    We show an English example just to explain the issue.
    EDS-NLP remains a **French-language** medical NLP library.

To curb this issue, EDS-NLP proposes rule-based pipes that qualify entities to help the user make an informed decision about which patient should be included in a real-world data cohort.

## Where do we get our spans ? {: #edsnlp.pipes.base.SpanGetterArg }

A component get entities from a document by looking up `doc.ents` or `doc.spans[group]`. This behavior is set by the `span_getter` argument in components that support it.

::: edsnlp.pipes.base.SpanGetterArg
    options:
        heading_level: 2
        show_bases: false
        show_source: false
        only_class_level: true

## Under the hood

Our _qualifier_ pipes all follow the same basic pattern:

1.  The pipeline extracts cues. We define three (possibly overlapping) kinds :

    - `preceding`, ie cues that _precede_ modulated entities ;
    - `following`, ie cues that _follow_ modulated entities ;
    - in some cases, `verbs`, ie verbs that convey a modulation (treated as preceding cues).

2.  The pipeline splits the text between sentences and propositions, using annotations from a sentencizer pipeline and `termination` patterns, which define syntagma/proposition terminations.

3.  For each pre-extracted entity, the pipeline checks whether there is a cue between the start of the syntagma and the start of the entity, or a following cue between the end of the entity and the end of the proposition.

Albeit simple, this algorithm can achieve very good performance depending on the modality. For instance, our `eds.negation` pipeline reaches 88% F1-score on our dataset.

!!! note "Dealing with pseudo-cues"

    The pipeline can also detect **pseudo-cues**, ie phrases that contain cues but **that are not cues themselves**. For instance: `sans doute`/`without doubt` contains `sans/without`, but does not convey negation.

    Detecting pseudo-cues lets the pipeline filter out any cue that overlaps with a pseudo-cue.

!!! warning "Sentence boundaries are required"

    The rule-based algorithm detects cues, and propagate their modulation on the rest of the [syntagma](https://en.wikipedia.org/wiki/Syntagma_(linguistics)){target=_blank}. For that reason, a qualifier pipeline needs a sentencizer component to be defined, and will fail otherwise.

    You may use EDS-NLP's:

    ```{ .python .no-check }
    nlp.add_pipe("eds.sentences")
    ```

## Persisting the results

Our qualifier pipelines write their results to a custom [spaCy extension](https://spacy.io/usage/processing-pipelines#custom-components-attributes){target=_blank}, defined on both `Span` and `Token` objects. We follow the convention of naming said attribute after the pipeline itself, eg `Span._.negation` for the`eds.negation` pipeline.

We also provide a string representation of the result, computed on the fly by declaring a getter that reads the boolean result of the pipeline. Following spaCy convention, we give this attribute the same name, followed by a `_`.
