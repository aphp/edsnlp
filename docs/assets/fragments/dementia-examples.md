=== "1"
    ```python
    text = "D'importants déficits cognitifs"
    doc = nlp(text)
    spans = doc.spans["dementia"]

    spans
    # Out: [déficits cognitifs]
    ```



=== "2"
    ```python
    text = "Patient atteint de démence"
    doc = nlp(text)
    spans = doc.spans["dementia"]

    spans
    # Out: [démence]
    ```



=== "3"
    ```python
    text = "On retrouve des anti-SLA"
    doc = nlp(text)
    spans = doc.spans["dementia"]

    spans
    # Out: []
    ```



=== "4"
    ```python
    text = "Une maladie de Charcot"
    doc = nlp(text)
    spans = doc.spans["dementia"]

    spans
    # Out: [maladie de Charcot]
    ```
