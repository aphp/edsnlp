=== "1"
    ```python
    text = "Patient autonome pour les activités de la vie quotidienne."
    doc = nlp(text)
    spans = doc.spans["autonomy"]

    spans
    # Out: ['autonome pour les activités da la vie quotidienne']

    span = spans[0]
    span._.autonomy
    # Out: 'healthy'
    ```



=== "2"
    ```python
    text = "La patiente sort de moins en moins"
    doc = nlp(text)
    spans = doc.spans["autonomy"]

    spans
    # Out: ['sort de moins en moins']

    span = spans[0]
    span._.autonomy
    # Out: 'altered_mild'
    ```
