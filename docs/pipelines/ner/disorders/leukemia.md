# Leukemia

The `eds.leukemia` pipeline component extracts mentions of leukemia.

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/disorders/leukemia/patterns.py"
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
nlp.add_pipe(f"eds.leukemia")
```

Below are a few examples:




=== "1"
    ```python
    text = "Sydrome myéloprolifératif"
    doc = nlp(text)
    spans = doc.spans["leukemia"]

    spans
    # Out: [myéloprolifératif]
    ```



=== "2"
    ```python
    text = "Sydrome myéloprolifératif bénin"
    doc = nlp(text)
    spans = doc.spans["leukemia"]

    spans
    # Out: []
    ```



=== "3"
    ```python
    text = "Patient atteint d'une LAM"
    doc = nlp(text)
    spans = doc.spans["leukemia"]

    spans
    # Out: [LAM]
    ```



=== "4"
    ```python
    text = "Une maladie de Vaquez"
    doc = nlp(text)
    spans = doc.spans["leukemia"]

    spans
    # Out: [Vaquez]
    ```

## Authors and citation

The `eds.leukemia` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
