=== "1"
    ```python
    text = "La patiente sort tous les jours."
    doc = nlp(text)
    spans = doc.spans["mobility_status"]

    spans
    # Out: [sort tous les jours]

    span = spans[0]
    span._.mobility_status
    # Out: 'healthy'
    ```

=== "2"
    ```python
    text = "Patient grabataire."
    doc = nlp(text)
    spans = doc.spans["mobility_status"]

    spans
    # Out: [grabataire]

    span = spans[0]
    span._.mobility_status
    # Out: 'altered_severe'
    ```

=== "3"
    ```python
    text = "Syndrome post-chute."
    doc = nlp(text)
    spans = doc.spans["mobility_status"]

    spans
    # Out: [Syndrome post-chute]

    span = spans[0]
    span._.mobility_status
    # Out: 'altered_mild'
    ```
