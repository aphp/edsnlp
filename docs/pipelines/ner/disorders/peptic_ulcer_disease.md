# Peptic ulcer disease

The `eds.peptic_ulcer_disease` pipeline component extracts mentions of peptic ulcer disease.

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/disorders/peptic_ulcer_disease/patterns.py"
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
nlp.add_pipe(f"eds.peptic_ulcer_disease")
```

Below are a few examples:




=== "1"
    ```python
    text = "Beaucoup d'ulcères gastriques"
    doc = nlp(text)
    spans = doc.spans["peptic_ulcer_disease"]

    spans
    # Out: [ulcères gastriques]
    ```



=== "2"
    ```python
    text = "Présence d'UGD"
    doc = nlp(text)
    spans = doc.spans["peptic_ulcer_disease"]

    spans
    # Out: [UGD]
    ```



=== "3"
    ```python
    text = "La patient à des ulcères"
    doc = nlp(text)
    spans = doc.spans["peptic_ulcer_disease"]

    spans
    # Out: []
    ```



=== "4"
    ```python
    text = "Au niveau gastrique: blabla blabla blabla blabla blabla quelques ulcères"
    doc = nlp(text)
    spans = doc.spans["peptic_ulcer_disease"]

    spans
    # Out: [ulcères]

    span = spans[0]

    span._.assigned
    # Out: {'is_peptic': [gastrique]}
    ```

## Authors and citation

The `eds.peptic_ulcer_disease` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
