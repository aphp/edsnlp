# Solid tumor

The `eds.solid_tumor` pipeline component extracts mentions of solid tumors. It will notably match:

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/disorders/solid_tumor/patterns.py"
    # fmt: on
    ```

## Extensions

On each span `span` that match, the following attributes are available:

- `span._.detailled_status`: set to either
    - `"METASTASIS"` for tumors at the metastatic stage
    - `"LOCALIZED"` else
- `span._.assigned`: dictionary with the following keys, if relevant:
    - `stage`: stage of the tumor

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
nlp.add_pipe(f"eds.solid_tumor")
```

Below are a few examples:




=== "1"
    ```python
    text = "Présence d'un carcinome intra-hépatique."
    doc = nlp(text)
    spans = doc.spans["solid_tumor"]

    spans
    # Out: [carcinome]
    ```



=== "2"
    ```python
    text = "Patient avec un K sein."
    doc = nlp(text)
    spans = doc.spans["solid_tumor"]

    spans
    # Out: [K sein]
    ```



=== "3"
    ```python
    text = "Il y a une tumeur bénigne"
    doc = nlp(text)
    spans = doc.spans["solid_tumor"]

    spans
    # Out: []
    ```



=== "4"
    ```python
    text = "Tumeur métastasée"
    doc = nlp(text)
    spans = doc.spans["solid_tumor"]

    spans
    # Out: [Tumeur métastasée]

    span = spans[0]

    span._.detailled_status
    # Out: METASTASIS

    span._.assigned
    # Out: {'metastasis': métastasée}
    ```



=== "5"
    ```python
    text = "Cancer du poumon au stade 4"
    doc = nlp(text)
    spans = doc.spans["solid_tumor"]

    spans
    # Out: [Cancer du poumon au stade 4]

    span = spans[0]

    span._.detailled_status
    # Out: METASTASIS

    span._.assigned
    # Out: {'stage': 4}
    ```



=== "6"
    ```python
    text = "Cancer du poumon au stade 2"
    doc = nlp(text)
    spans = doc.spans["solid_tumor"]

    spans
    # Out: [Cancer du poumon au stade 2]

    span = spans[0]

    span._.assigned
    # Out: {'stage': 2}
    ```



=== "7"
    ```python
    text = "Présence de nombreuses lésions secondaires"
    doc = nlp(text)
    spans = doc.spans["solid_tumor"]

    spans
    # Out: [lésions secondaires]

    span = spans[0]

    span._.detailled_status
    # Out: METASTASIS
    ```

## Authors and citation

The `eds.solid_tumor` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
