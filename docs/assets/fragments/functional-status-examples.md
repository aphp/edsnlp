=== "1"
    ```python
    text = "Patient autonome pour les activités de la vie quotidienne."
    doc = nlp(text)
    spans = doc.spans["functional_status"]

    spans
    # Out: [autonome pour les activités de la vie quotidienne]

    span = spans[0]
    span._.functional_status
    # Out: 'healthy'
    ```



=== "2"
    ```python
    text = "La patiente sort de moins en moins"
    doc = nlp(text)
    spans = doc.spans["functional_status"]

    spans
    # Out: [sort de moins en moins]

    span = spans[0]
    span._.functional_status
    # Out: 'altered_mild'
    ```
