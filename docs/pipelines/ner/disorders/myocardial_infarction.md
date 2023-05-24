# Myocardial infarction

The `eds.myocardial_infarction` pipeline component extracts mentions of myocardial infarction. It will notably match:

- Mentions of various diseases (see below)
- Mentions of stents with a heart localization

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/disorders/myocardial_infarction/patterns.py"
    # fmt: on
    ```

## Extensions

On each span `span` that match, the following attributes are available:

- `span._.detailled_status`: set to `"PRESENT"`
- `span._.assigned`: dictionary with the following keys, if relevant:
    - `heart_localized`: localization of the stent or bypass

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
nlp.add_pipe(f"eds.myocardial_infarction")
```

Below are a few examples:




=== "1"
    ```python
    text = "Une cardiopathie ischémique"
    doc = nlp(text)
    spans = doc.spans["myocardial_infarction"]

    spans
    # Out: [cardiopathie ischémique]
    ```



=== "2"
    ```python
    text = "Une cardiopathie non-ischémique"
    doc = nlp(text)
    spans = doc.spans["myocardial_infarction"]

    spans
    # Out: []
    ```



=== "3"
    ```python
    text = "Présence d'un stent sur la marginale"
    doc = nlp(text)
    spans = doc.spans["myocardial_infarction"]

    spans
    # Out: [stent sur la marginale]

    span = spans[0]

    span._.assigned
    # Out: {'heart_localized': [marginale]}
    ```



=== "4"
    ```python
    text = "Présence d'un stent périphérique"
    doc = nlp(text)
    spans = doc.spans["myocardial_infarction"]

    spans
    # Out: []
    ```



=== "5"
    ```python
    text = "infarctus du myocarde"
    doc = nlp(text)
    spans = doc.spans["myocardial_infarction"]

    spans
    # Out: [infarctus du myocarde]

    span = spans[0]

    span._.assigned
    # Out: {'heart_localized': [myocarde]}
    ```

## Authors and citation

The `eds.myocardial_infarction` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
