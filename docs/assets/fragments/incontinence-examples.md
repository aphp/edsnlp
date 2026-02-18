=== "1"
    ```python
    text = "Patient continent."
    doc = nlp(text)
    spans = doc.spans["incontinence"]

    spans
    # Out: ['continent']

    span = spans[0]
    span._.incontinence
    # Out: 'healthy'
    ```

=== "2"
    ```python
    text = "Installation d'une sonde à demeure."
    doc = nlp(text)
    spans = doc.spans["incontinence"]

    spans
    # Out: ['sonde à deumeure']

    span = spans[0]
    span._.incontinence
    # Out: 'altered_severe'
    ```
