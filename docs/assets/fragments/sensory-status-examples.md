=== "1"
    ```python
    text = "Patiente sourde"
    doc = nlp(text)
    spans = doc.spans["sensory_status"]

    spans
    # Out: [sourde]

    span = spans[0]
    span._.sensory_status
    # Out: 'altered_severe'
    ```

=== "2"
    ```python
    text = "Coeur sourd"
    doc = nlp(text)
    spans = doc.spans["sensory_status"]

    spans
    # Out: []
    ```
