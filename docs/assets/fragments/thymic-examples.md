=== "1"
    ```python
    text = "Le patient est anxieux"
    doc = nlp(text)
    spans = doc.spans["thymic"]

    spans
    # Out: ['anxieux']

    span = spans[0]
    span._.thymic
    # Out: 'altered_nondescript'
    ```

=== "2"
    ```python
    text = "Le moral est bon"
    doc = nlp(text)
    spans = doc.spans["thymic"]

    spans
    # Out: ['moral est bon']

    span = spans[0]
    span._.thymic
    # Out: 'healthy'
    ```
