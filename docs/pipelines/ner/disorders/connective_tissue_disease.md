# Connective tissue disease

The `eds.connective_tissue_disease` pipeline component extracts mentions of connective tissue diseases.

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/disorders/connective_tissue_disease/patterns.py"
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
nlp.add_pipe(f"eds.connective_tissue_disease")
```

Below are a few examples:




=== "1"
    ```python
    text = "Présence d'une sclérodermie."
    doc = nlp(text)
    spans = doc.spans["connective_tissue_disease"]

    spans
    # Out: [sclérodermie]
    ```



=== "2"
    ```python
    text = "Patient atteint d'un lupus."
    doc = nlp(text)
    spans = doc.spans["connective_tissue_disease"]

    spans
    # Out: [lupus]
    ```



=== "3"
    ```python
    text = "Présence d'anticoagulants lupiques,"
    doc = nlp(text)
    spans = doc.spans["connective_tissue_disease"]

    spans
    # Out: []
    ```



=== "4"
    ```python
    text = "Il y a une MICI."
    doc = nlp(text)
    spans = doc.spans["connective_tissue_disease"]

    spans
    # Out: [MICI]
    ```



=== "5"
    ```python
    text = "Syndrome de Raynaud"
    doc = nlp(text)
    spans = doc.spans["connective_tissue_disease"]

    spans
    # Out: [Raynaud]
    ```

## Authors and citation

The `eds.connective_tissue_disease` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
