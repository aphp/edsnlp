=== "1"
    ```python
    text = "Sur le plan antalgique, rien à signaler."
    doc = nlp(text)
    spans = doc.spans["pain"]

    spans
    # Out: ['Sur le plan antalgique']

    span = spans[0]
    span._.pain
    # Out: 'other'
    ```

=== "2"
    ```python
    text = "palier 1"
    doc = nlp(text)
    spans = doc.spans["pain"]

    spans
    # Out: ['palier 1']

    span = spans[0]
    span._.pain
    # Out: 'altered_mild'
    ```
