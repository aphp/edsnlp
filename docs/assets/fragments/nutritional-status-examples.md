=== "1"
    ```python
    text = "Carence en vitamines B9."
    doc = nlp(text)
    spans = doc.spans["nutritional_status"]

    spans
    # Out: [Carence en vitamines B9]

    span = spans[0]
    span._.nutritional_status
    # Out: 'altered_nondescript'
    ```

=== "2"
    ```python
    text = "Vitamines B9."
    doc = nlp(text)
    spans = doc.spans["nutritional_status"]

    spans
    # Out: []
    ```
