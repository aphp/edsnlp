=== "1"
    ```python
    text = "Patiente sourde"
    doc = nlp(text)
    spans = doc.spans["sensory"]

    spans
    # Out: ['sourde']

    span = spans[0]
    span._.sensory
    # Out: 'altered_severe'
    ```

=== "2"
    ```python
    text = "Coeur sourd"
    doc = nlp(text)
    spans = doc.spans["sensory"]

    spans
    # Out: []
    ```
