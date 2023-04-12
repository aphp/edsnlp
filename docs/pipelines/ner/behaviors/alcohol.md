# Alcohol consumption

The `eds.alcohol` pipeline component extracts mentions of alcohol consumption. It won't match occasionnal consumption, nor acute intoxication.

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/behaviors/alcohol/patterns.py"
    # fmt: on
    ```

## Extensions

On each span `span` that match, the following attributes are available:

- `span._.status_`: set to either
    - `"PRESENT"`
    - `"ABSTINENCE"` if the patient stopped its consumption
    - `"ABSENT"` if the patient has no alcohol dependence

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
nlp.add_pipe(f"eds.alcohol")
```

Below are a few examples:




=== "1"
    ```python
    text = "Patient alcoolique."
    doc = nlp(text)
    spans = doc.spans["alcohol"]

    spans
    # Out: [alcoolique]
    ```



=== "2"
    ```python
    text = "OH chronique."
    doc = nlp(text)
    spans = doc.spans["alcohol"]

    spans
    # Out: [OH]
    ```



=== "3"
    ```python
    text = "Prise d'alcool occasionnelle"
    doc = nlp(text)
    spans = doc.spans["alcohol"]

    spans
    # Out: []
    ```



=== "4"
    ```python
    text = "Application d'un pansement alcoolisé"
    doc = nlp(text)
    spans = doc.spans["alcohol"]

    spans
    # Out: []
    ```



=== "5"
    ```python
    text = "Alcoolisme sevré"
    doc = nlp(text)
    spans = doc.spans["alcohol"]

    spans
    # Out: [Alcoolisme sevré]

    span = spans[0]

    span._.status_
    # Out: ABSTINENCE

    span._.assigned
    # Out: {'stopped': [sevré]}
    ```



=== "6"
    ```python
    text = "Alcoolisme non sevré"
    doc = nlp(text)
    spans = doc.spans["alcohol"]

    spans
    # Out: [Alcoolisme]
    ```



=== "7"
    ```python
    text = "Alcool: 0"
    doc = nlp(text)
    spans = doc.spans["alcohol"]

    spans
    # Out: [Alcool: 0]

    span = spans[0]

    span._.status_
    # Out: ABSENT

    span._.assigned
    # Out: {'zero_after': [0]}
    ```



=== "8"
    ```python
    text = "Le patient est en cours de sevrage éthylotabagique"
    doc = nlp(text)
    spans = doc.spans["alcohol"]

    spans
    # Out: [sevrage éthylotabagique]

    span = spans[0]

    span._.status_
    # Out: ABSTINENCE

    span._.assigned
    # Out: {'stopped': [sevrage]}
    ```

## Authors and citation

The `eds.alcohol` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
