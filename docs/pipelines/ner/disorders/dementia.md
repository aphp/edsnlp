# Dementia

The `eds.dementia` pipeline component extracts mentions of dementia.

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/disorders/dementia/patterns.py"
    # fmt: on
    ```

## Extensions

On each span `span` that match, the following attributes are available:

- `span._.status_`: set to `"PRESENT"`

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
nlp.add_pipe(f"eds.dementia")
```

Below are a few examples:




=== "1"
    ```python
    text = "D'importants déficits cognitifs"
    doc = nlp(text)
    spans = doc.spans["dementia"]

    spans
    # Out: [déficits cognitifs]
    ```



=== "2"
    ```python
    text = "Patient atteint de démence"
    doc = nlp(text)
    spans = doc.spans["dementia"]

    spans
    # Out: [démence]
    ```



=== "3"
    ```python
    text = "On retrouve des anti-SLA"
    doc = nlp(text)
    spans = doc.spans["dementia"]

    spans
    # Out: []
    ```



=== "4"
    ```python
    text = "Une maladie de Charcot"
    doc = nlp(text)
    spans = doc.spans["dementia"]

    spans
    # Out: [maladie de Charcot]
    ```

## Authors and citation

The `eds.dementia` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
