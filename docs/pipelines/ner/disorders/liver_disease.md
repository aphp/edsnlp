# Liver disease

The `eds.liver_disease` pipeline component extracts mentions of liver disease.

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/disorders/liver_disease/patterns.py"
    # fmt: on
    ```

## Extensions

On each span `span` that match, the following attributes are available:

- `span._.status_`: set to either
    - `"MILD"` for mild liver diseases
    - `"MODERATE_TO_SEVERE"` else

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
nlp.add_pipe(f"eds.liver_disease")
```

Below are a few examples:




=== "1"
    ```python
    text = "Il y a une fibrose hépatique"
    doc = nlp(text)
    spans = doc.spans["liver_disease"]

    spans
    # Out: [fibrose hépatique]
    ```



=== "2"
    ```python
    text = "Une hépatite B chronique"
    doc = nlp(text)
    spans = doc.spans["liver_disease"]

    spans
    # Out: [hépatite B chronique]
    ```



=== "3"
    ```python
    text = "Le patient consulte pour une cirrhose"
    doc = nlp(text)
    spans = doc.spans["liver_disease"]

    spans
    # Out: [cirrhose]

    span = spans[0]

    span._.status_
    # Out: MODERATE_TO_SEVERE
    ```



=== "4"
    ```python
    text = "Greffe hépatique."
    doc = nlp(text)
    spans = doc.spans["liver_disease"]

    spans
    # Out: [Greffe hépatique]

    span = spans[0]

    span._.status_
    # Out: MODERATE_TO_SEVERE
    ```

## Authors and citation

The `eds.liver_disease` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
