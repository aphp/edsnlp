# Tobacco consumption

The `eds.tobacco` pipeline component extracts mentions of tobacco consumption.

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/behaviors/tobacco/patterns.py"
    # fmt: on
    ```

## Extensions

On each span `span` that match, the following attributes are available:

- `span._.status_`: set to either
    - `"PRESENT"`
    - `"ABSTINENCE"` if the patient stopped its consumption
    - `"ABSENT"` if the patient has no tobacco dependence
- `span._.assigned`: dictionary with the following keys, if relevant:
    - `PA`: the mentionned *year-pack* (= *paquet-année*)
    - `secondhand`: if secondhand smoking

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
nlp.add_pipe(f"eds.tobacco")
```

Below are a few examples:




=== "1"
    ```python
    text = "Tabagisme évalué à 15 PA"
    doc = nlp(text)
    spans = doc.spans["tobacco"]

    spans
    # Out: [Tabagisme évalué à 15 PA]

    span = spans[0]

    span._.assigned
    # Out: {'PA': 15}
    ```



=== "2"
    ```python
    text = "Patient tabagique"
    doc = nlp(text)
    spans = doc.spans["tobacco"]

    spans
    # Out: [tabagique]
    ```



=== "3"
    ```python
    text = "Tabagisme festif"
    doc = nlp(text)
    spans = doc.spans["tobacco"]

    spans
    # Out: []
    ```



=== "4"
    ```python
    text = "On a un tabagisme ancien"
    doc = nlp(text)
    spans = doc.spans["tobacco"]

    spans
    # Out: [tabagisme ancien]

    span = spans[0]

    span._.status_
    # Out: ABSTINENCE

    span._.assigned
    # Out: {'stopped': [ancien]}
    ```



=== "5"
    ```python
    text = "Tabac: 0"
    doc = nlp(text)
    spans = doc.spans["tobacco"]

    spans
    # Out: [Tabac: 0]

    span = spans[0]

    span._.status_
    # Out: ABSENT

    span._.assigned
    # Out: {'zero_after': [0]}
    ```



=== "6"
    ```python
    text = "Tabagisme passif"
    doc = nlp(text)
    spans = doc.spans["tobacco"]

    spans
    # Out: [Tabagisme passif]

    span = spans[0]

    span._.status_
    # Out: ABSENT

    span._.assigned
    # Out: {'secondhand': passif}
    ```



=== "7"
    ```python
    text = "Tabac: sevré depuis 5 ans"
    doc = nlp(text)
    spans = doc.spans["tobacco"]

    spans
    # Out: [Tabac: sevré]

    span = spans[0]

    span._.status_
    # Out: ABSTINENCE

    span._.assigned
    # Out: {'stopped': [sevré]}
    ```

## Authors and citation

The `eds.tobacco` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
