# Peripheral vascular disease

The `eds.peripheral_vascular_disease` pipeline component extracts mentions of peripheral vascular disease.

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/disorders/peripheral_vascular_disease/patterns.py"
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
nlp.add_pipe(f"eds.peripheral_vascular_disease")
```

Below are a few examples:




=== "1"
    ```python
    text = "Un AOMI"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: [AOMI]
    ```



=== "2"
    ```python
    text = "Présence d'un infarctus rénal"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: [infarctus rénal]
    ```



=== "3"
    ```python
    text = "Une angiopathie cérébrale"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: []
    ```



=== "4"
    ```python
    text = "Une angiopathie"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: [angiopathie]
    ```



=== "5"
    ```python
    text = "Une thrombose cérébrale"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: []
    ```



=== "6"
    ```python
    text = "Une thrombose des veines superficielles"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: []
    ```



=== "7"
    ```python
    text = "Une thrombose"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: [thrombose]
    ```



=== "8"
    ```python
    text = "Effectuer un bilan pre-trombose"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: []
    ```



=== "9"
    ```python
    text = "Une ischémie des MI est remarquée."
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: [ischémie des MI]

    span = spans[0]

    span._.assigned
    # Out: {'peripheral': [MI]}
    ```



=== "10"
    ```python
    text = "Plusieurs cas d'EP"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: [EP]
    ```



=== "11"
    ```python
    text = "Effectuer des cures d'EP"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: []
    ```



=== "12"
    ```python
    text = "Le patient est hypertendu"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: [hypertendu]
    ```



=== "13"
    ```python
    text = "Une hypertension portale"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: []
    ```

## Authors and citation

The `eds.peripheral_vascular_disease` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
