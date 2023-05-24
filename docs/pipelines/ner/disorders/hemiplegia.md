# Hemiplegia

The `eds.hemiplegia` pipeline component extracts mentions of hemiplegia.

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/disorders/hemiplegia/patterns.py"
    # fmt: on
    ```


## Extensions

On each span `span` that match, the following attributes are available:

- `span._.detailled_status`: set to `"PRESENT"`

## Usage

```python
import spacy

nlp = spacy.blank("eds")
nlp.add_pipe("eds.sentences")
nlp.add_pipe(
    "eds.normalizer",
    config=dict(
        accents=True,
        lowercase=True,
        quotes=True,
        spaces=True,
        pollution=dict(
            information=True,
            bars=True,
            biology=True,
            doctors=True,
            web=True,
            coding=True,
            footer=True,
        ),
    ),
)
nlp.add_pipe(f"eds.hemiplegia")
```

Below are a few examples:




=== "1"
    ```python
    text = "Patient hémiplégique"
    doc = nlp(text)
    spans = doc.spans["hemiplegia"]

    spans
    # Out: [hémiplégique]
    ```



=== "2"
    ```python
    text = "Paralysie des membres inférieurs"
    doc = nlp(text)
    spans = doc.spans["hemiplegia"]

    spans
    # Out: [Paralysie des membres]
    ```



=== "3"
    ```python
    text = "Patient en LIS"
    doc = nlp(text)
    spans = doc.spans["hemiplegia"]

    spans
    # Out: [LIS]
    ```

## Authors and citation

The `eds.hemiplegia` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
