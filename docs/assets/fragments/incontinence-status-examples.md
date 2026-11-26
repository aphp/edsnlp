=== "1"
    ```python
    text = "Patient continent."
    doc = nlp(text)
    spans = doc.spans["incontinence_status"]

    spans
    # Out: [continent]

    span = spans[0]
    span._.incontinence_status
    # Out: 'healthy'
    ```

=== "2"
    ```python
    text = "Installation d'une sonde à demeure."
    doc = nlp(text)
    spans = doc.spans["incontinence_status"]

    spans
    # Out: [sonde à demeure]

    span = spans[0]
    span._.incontinence_status
    # Out: 'altered_severe'
    ```
