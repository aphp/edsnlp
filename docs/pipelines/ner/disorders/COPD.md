# COPD

The `eds.COPD` pipeline component extracts mentions of COPD (*Chronic obstructive pulmonary disease*). It will notably match:

- Mentions of various diseases (see below)
- Pulmonary hypertension
- Long-term oxygen therapy

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/disorders/COPD/patterns.py"
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
nlp.add_pipe(f"eds.COPD")
```

Below are a few examples:




=== "1"
    ```python
    text = "Une fibrose interstitielle diffuse idiopathique"
    doc = nlp(text)
    spans = doc.spans["COPD"]

    spans
    # Out: [fibrose interstitielle diffuse idiopathique]
    ```



=== "2"
    ```python
    text = "Patient atteint de pneumoconiose"
    doc = nlp(text)
    spans = doc.spans["COPD"]

    spans
    # Out: [pneumoconiose]
    ```



=== "3"
    ```python
    text = "Présence d'une HTAP."
    doc = nlp(text)
    spans = doc.spans["COPD"]

    spans
    # Out: [HTAP]
    ```



=== "4"
    ```python
    text = "On voit une hypertension pulmonaire minime"
    doc = nlp(text)
    spans = doc.spans["COPD"]

    spans
    # Out: []
    ```



=== "5"
    ```python
    text = "La patiente a été mis sous oxygénorequérance"
    doc = nlp(text)
    spans = doc.spans["COPD"]

    spans
    # Out: []
    ```



=== "6"
    ```python
    text = "La patiente est sous oxygénorequérance au long cours"
    doc = nlp(text)
    spans = doc.spans["COPD"]

    spans
    # Out: [oxygénorequérance au long cours]

    span = spans[0]

    span._.assigned
    # Out: {'long': [long cours]}
    ```

## Authors and citation

The `eds.COPD` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
