# Diabetes

The `eds.diabetes` pipeline component extracts mentions of diabetes.

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/disorders/diabetes/patterns.py"
    # fmt: on
    ```

## Extensions

On each span `span` that match, the following attributes are available:

- `span._.detailled_status`: set to either
    - `"WITH_COMPLICATION"` if the diabetes is  complicated (e.g., via organ damages)
    - `"WITHOUT_COMPLICATION"` else
- `span._.assigned`: dictionary with the following keys, if relevant:
    - `type`: type of diabetes (I or II)
    - `insulin`: if the diabetes is insulin-dependent
    - `cortico`: if the diabetes if corticoid-induced

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
nlp.add_pipe(f"eds.diabetes")
```

Below are a few examples:




=== "1"
    ```python
    text = "Présence d'un DT2"
    doc = nlp(text)
    spans = doc.spans["diabetes"]

    spans
    # Out: [DT2]
    ```



=== "2"
    ```python
    text = "Présence d'un DNID"
    doc = nlp(text)
    spans = doc.spans["diabetes"]

    spans
    # Out: [DNID]
    ```



=== "3"
    ```python
    text = "Patient diabétique"
    doc = nlp(text)
    spans = doc.spans["diabetes"]

    spans
    # Out: [diabétique]
    ```



=== "4"
    ```python
    text = "Un diabète insipide"
    doc = nlp(text)
    spans = doc.spans["diabetes"]

    spans
    # Out: []
    ```



=== "5"
    ```python
    text = "Atteinte neurologique d'origine diabétique"
    doc = nlp(text)
    spans = doc.spans["diabetes"]

    spans
    # Out: [origine diabétique]

    span = spans[0]

    span._.detailled_status
    # Out: WITH_COMPLICATION

    span._.assigned
    # Out: {'complicated_before': [origine]}
    ```



=== "6"
    ```python
    text = "Une rétinopathie diabétique"
    doc = nlp(text)
    spans = doc.spans["diabetes"]

    spans
    # Out: [rétinopathie diabétique]

    span = spans[0]

    span._.detailled_status
    # Out: WITH_COMPLICATION

    span._.assigned
    # Out: {'complicated_before': [rétinopathie]}
    ```



=== "7"
    ```python
    text = "Il y a un mal perforant plantaire"
    doc = nlp(text)
    spans = doc.spans["diabetes"]

    spans
    # Out: [mal perforant plantaire]

    span = spans[0]

    span._.detailled_status
    # Out: WITH_COMPLICATION
    ```

## Authors and citation

The `eds.diabetes` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
