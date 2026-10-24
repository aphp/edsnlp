=== "1"
    ```python
    text = "Patient trop fragile pour une opération."
    doc = nlp(text)
    spans = doc.spans["frailty_mentions"]

    spans
    # Out: [trop fragile]

    span = spans[0]
    span._.frailty_mentions
    # Out: 'altered_severe'
    ```



=== "2"
    ```python
    text = "Le patient présente une fragilité thymique."
    doc = nlp(text)
    spans = doc.spans["frailty_mentions"]

    spans
    # Out: []
    ```
