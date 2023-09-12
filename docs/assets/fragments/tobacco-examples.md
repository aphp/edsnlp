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

    span._.detailed_status
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

    span._.detailed_status
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

    span._.detailed_status
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

    span._.detailed_status
    # Out: ABSTINENCE

    span._.assigned
    # Out: {'stopped': [sevré]}
    ```
