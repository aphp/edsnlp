=== "1"
    ```python
    text = "Le patient est anxieux"
    doc = nlp(text)
    spans = doc.spans["psychological_status"]

    spans
    # Out: [anxieux]

    span = spans[0]
    span._.psychological_status
    # Out: 'altered_nondescript'
    ```

=== "2"
    ```python
    text = "Le moral est bon"
    doc = nlp(text)
    spans = doc.spans["psychological_status"]

    spans
    # Out: [moral est bon]

    span = spans[0]
    span._.psychological_status
    # Out: 'healthy'
    ```
