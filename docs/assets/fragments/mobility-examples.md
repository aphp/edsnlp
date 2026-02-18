=== "1"
    ```python
    text = "La patiente sort tous les jours."
    doc = nlp(text)
    spans = doc.spans["mobility"]

    spans
    # Out: ['sort tous les jours']

    span = spans[0]
    span._.mobility
    # Out: 'healthy'
    ```

=== "2"
    ```python
    text = "Patient grabataire."
    doc = nlp(text)
    spans = doc.spans["mobility"]

    spans
    # Out: ['grabataire']

    span = spans[0]
    span._.mobility
    # Out: 'altered_severe'
    ```

=== "3"
    ```python
    text = "Syndrome post-chute."
    doc = nlp(text)
    spans = doc.spans["mobility"]

    spans
    # Out: ['Syndrome post-chute']

    span = spans[0]
    span._.mobility
    # Out: 'altered_mild'
    ```
