=== "1"
    ```python
    text = "On constate une iatrogénie"
    doc = nlp(text)
    spans = doc.spans["polymed"]

    spans
    # Out: ['iatrogénie']

    span = spans[0]
    span._.pain
    # Out: 'altered_nondescript'
    ```
