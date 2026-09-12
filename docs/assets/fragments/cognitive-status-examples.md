=== "1"
    ```python
    text = "Le patient a un langage fluent et cohérent"
    doc = nlp(text)
    spans = doc.spans["cognitive_status"]

    spans
    # Out: [langage fluent]

    span = spans[0]
    span._.cognitive_status
    # Out: 'healthy'
    ```



=== "2"
    ```python
    text = "Période de confusion aiguë"
    doc = nlp(text)
    spans = doc.spans["cognitive_status"]

    spans
    # Out: []
    ```



=== "3"
    ```python
    text = "Signes de démence"
    doc = nlp(text)
    spans = doc.spans["cognitive_status"]

    spans
    # Out: [démence]

    span = spans[0]
    span._.cognitive_status
    # Out: 'altered_severe'
    ```
