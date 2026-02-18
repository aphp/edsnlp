=== "1"
    ```python
    text = "Patient trop fragile pour une opération."
    doc = nlp(text)
    spans = doc.spans["frailty"]

    spans
    # Out: ['trop fragile']

    span = spans[0]
    span._.frailty
    # Out: 'altered_severe'
    ```



=== "2"
    ```python
    text = "Le patient présente une fragilité thymique."
    doc = nlp(text)
    spans = doc.spans["frailty"]

    spans
    # Out: []
    ```
