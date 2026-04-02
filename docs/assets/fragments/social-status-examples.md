=== "1"
    ```python
    text = "Patiente peu entourée"
    doc = nlp(text)
    spans = doc.spans["social_status"]

    spans
    # Out: [peu entourée]

    span = spans[0]
    span._.social_status
    # Out: 'altered_nondescript'
    ```

=== "2"
    ```python
    text = "Le patient vit avec son épouse"
    doc = nlp(text)
    spans = doc.spans["social_status"]

    spans
    # Out: [vit avec son épouse]

    span = spans[0]
    span._.social_status
    # Out: 'healthy'
    ```
