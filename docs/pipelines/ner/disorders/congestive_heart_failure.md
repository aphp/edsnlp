# Congestive heart failure

The `eds.congestive_heart_failure` pipeline component extracts mentions of congestive heart failure. It will notably match:

- Mentions of various diseases (see below)
- Heart transplantation
- AF (Atrial Fibrilation)
- Pace maker

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/disorders/congestive_heart_failure/patterns.py"
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
nlp.add_pipe(f"eds.congestive_heart_failure")
```

Below are a few examples:




=== "1"
    ```python
    text = "Présence d'un oedème pulmonaire"
    doc = nlp(text)
    spans = doc.spans["congestive_heart_failure"]

    spans
    # Out: [oedème pulmonaire]
    ```



=== "2"
    ```python
    text = "Le patient est équipé d'un pace-maker"
    doc = nlp(text)
    spans = doc.spans["congestive_heart_failure"]

    spans
    # Out: [pace-maker]
    ```



=== "3"
    ```python
    text = "Un cardiopathie non décompensée"
    doc = nlp(text)
    spans = doc.spans["congestive_heart_failure"]

    spans
    # Out: []
    ```



=== "4"
    ```python
    text = "Insuffisance cardiaque"
    doc = nlp(text)
    spans = doc.spans["congestive_heart_failure"]

    spans
    # Out: [Insuffisance cardiaque]
    ```



=== "5"
    ```python
    text = "Insuffisance cardiaque minime"
    doc = nlp(text)
    spans = doc.spans["congestive_heart_failure"]

    spans
    # Out: []
    ```

## Authors and citation

The `eds.congestive_heart_failure` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
