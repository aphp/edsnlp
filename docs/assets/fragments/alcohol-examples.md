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

    span._.detailed_status
    # Out: ABSTINENCE

    span._.assigned
    # Out: {'stopped': sevré}
    ```



=== "6"
    ```python
    text = "Alcoolisme non sevré"
    doc = nlp(text)
    spans = doc.spans["alcohol"]

    spans
    # Out: [Alcoolism non sevré]

    span = spans[0]

    span._.detailed_status
    # Out: None # "sevré" is negated, so no "ABTINENCE" status
    ```



=== "7"
    ```python
    text = "Alcool: 0"
    doc = nlp(text)
    spans = doc.spans["alcohol"]

    spans
    # Out: [Alcool: 0]

    span = spans[0]

    span._.negation
    # Out: True

    span._.assigned
    # Out: {'zero_after': 0}
    ```



=== "8"
    ```python
    text = "Le patient est en cours de sevrage éthylotabagique"
    doc = nlp(text)
    spans = doc.spans["alcohol"]

    spans
    # Out: [sevrage éthylotabagique]

    span = spans[0]

    span._.detailed_status
    # Out: ABSTINENCE

    span._.assigned
    # Out: {'stopped': sevrage}
    ```
