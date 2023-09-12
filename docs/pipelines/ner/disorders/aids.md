# AIDS

The `eds.aids` pipeline component extracts mentions of AIDS. It will notably match:

- Mentions of VIH/HIV at the SIDA/AIDS stage
- Mentions of VIH/HIV with opportunistic(s) infection(s)

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/disorders/AIDS/patterns.py"
    # fmt: on
    ```

!!! warning "On HIV infection"

    pre-AIDS HIV infection are not extracted, only AIDS.

## Extensions

On each span `span` that match, the following attributes are available:

- `span._.detailled_status`: set to `"PRESENT"`
- `span._.assigned`: dictionary with the following keys, if relevant:
    - `opportunist`: list of opportunist infections extracted around the HIV mention
    - `stage`: stage of the HIV infection


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
nlp.add_pipe(f"eds.aids")
```

Below are a few examples:

=== "SIDA"
    ```python
    text = "Patient atteint du VIH au stade SIDA."
    doc = nlp(text)
    spans = doc.spans["aids"]

    spans
    # Out: [VIH au stade SIDA]
    ```



=== "VIH"
    ```python
    text = "Patient atteint du VIH."
    doc = nlp(text)
    spans = doc.spans["aids"]

    spans
    # Out: []
    ```



=== "Coinfection"
    ```python
    text = "Il y a un VIH avec coinfection pneumocystose"
    doc = nlp(text)
    spans = doc.spans["aids"]

    spans
    # Out: [VIH]

    span = spans[0]

    span._.assigned
    # Out: {'opportunist': [coinfection, pneumocystose]}
    ```



=== "VIH stade SIDA"
    ```python
    text = "Présence d'un VIH stade C"
    doc = nlp(text)
    spans = doc.spans["aids"]

    spans
    # Out: [VIH]

    span = spans[0]

    span._.assigned
    # Out: {'stage': [C]}
    ```

## Authors and citation

The `eds.aids` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
