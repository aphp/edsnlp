# Cerebrovascular accident

The `eds.cerebrovascular_accident` pipeline component extracts mentions of cerebrovascular accident. It will notably match:

- Mentions of AVC/AIT
- Mentions of bleeding, hemorrhage, thrombus, ischemia, etc., localized in the brain

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/disorders/cerebrovascular_accident/patterns.py"
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
nlp.add_pipe(f"eds.cerebrovascular_accident")
```

Below are a few examples:




=== "1"
    ```python
    text = "Patient hospitalisé à AVC."
    doc = nlp(text)
    spans = doc.spans["cerebrovascular_accident"]

    spans
    # Out: []
    ```



=== "2"
    ```python
    text = "Hospitalisation pour un AVC."
    doc = nlp(text)
    spans = doc.spans["cerebrovascular_accident"]

    spans
    # Out: [AVC]
    ```



=== "3"
    ```python
    text = "Saignement intracranien"
    doc = nlp(text)
    spans = doc.spans["cerebrovascular_accident"]

    spans
    # Out: [Saignement]

    span = spans[0]

    span._.assigned
    # Out: {'brain_localized': [intracranien]}
    ```



=== "4"
    ```python
    text = "Thrombose périphérique"
    doc = nlp(text)
    spans = doc.spans["cerebrovascular_accident"]

    spans
    # Out: []
    ```



=== "5"
    ```python
    text = "Thrombose sylvienne"
    doc = nlp(text)
    spans = doc.spans["cerebrovascular_accident"]

    spans
    # Out: [Thrombose]

    span = spans[0]

    span._.assigned
    # Out: {'brain_localized': [sylvienne]}
    ```



=== "6"
    ```python
    text = "Infarctus cérébral"
    doc = nlp(text)
    spans = doc.spans["cerebrovascular_accident"]

    spans
    # Out: [Infarctus]

    span = spans[0]

    span._.assigned
    # Out: {'brain_localized': [cérébral]}
    ```



=== "7"
    ```python
    text = "Soigné via un thrombolyse"
    doc = nlp(text)
    spans = doc.spans["cerebrovascular_accident"]

    spans
    # Out: [thrombolyse]
    ```

## Authors and citation

The `eds.cerebrovascular_accident` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
