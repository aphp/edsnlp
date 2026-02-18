=== "1"
    ```python
    text = "Carence en vitamines B9."
    doc = nlp(text)
    spans = doc.spans["nutrition"]

    spans
    # Out: ['Carence en vitamines B9']

    span = spans[0]
    span._.nutrition
    # Out: 'altered_nondescript'
    ```

=== "2"
    ```python
    text = "Vitamines B9."
    doc = nlp(text)
    spans = doc.spans["nutrition"]

    spans
    # Out: []
    ```
