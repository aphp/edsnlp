# Lymphoma

The `eds.lymphoma` pipeline component extracts mentions of lymphoma.

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/disorders/lymphoma/patterns.py"
    # fmt: on
    ```

## Extensions

On each span `span` that match, the following attributes are available:

- `span._.detailled_status`: set to `"PRESENT"`

!!! warning "Monoclonal gammapathy"

    Monoclonal gammapathies are not extracted by this pipeline

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
nlp.add_pipe(f"eds.lymphoma")
```

Below are a few examples:




=== "1"
    ```python
    text = "Un lymphome de Hodgkin."
    doc = nlp(text)
    spans = doc.spans["lymphoma"]

    spans
    # Out: [lymphome de Hodgkin]
    ```



=== "2"
    ```python
    text = "Atteint d'un Waldenstörm"
    doc = nlp(text)
    spans = doc.spans["lymphoma"]

    spans
    # Out: [Waldenstörm]
    ```



=== "3"
    ```python
    text = "Un LAGC"
    doc = nlp(text)
    spans = doc.spans["lymphoma"]

    spans
    # Out: [LAGC]
    ```



=== "4"
    ```python
    text = "anti LAGC: 10^4/mL"
    doc = nlp(text)
    spans = doc.spans["lymphoma"]

    spans
    # Out: []
    ```

## Authors and citation

The `eds.lymphoma` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
