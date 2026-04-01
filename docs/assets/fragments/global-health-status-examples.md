=== "1"
    ```python
    text = "Patient trop fragile"
    doc = nlp(text)
    spans = doc.spans["global_health_status"]

    spans
    # Out: [trop fragile]

    span = spans[0]
    span._.global_health_status
    # Out: 'altered_severe'
    ```



=== "2"
    ```python
    text = "Le patient a une bonne qualité de vie."
    doc = nlp(text)
    spans = doc.spans["global_health_status"]

    spans
    # Out: []
    ```



=== "3"
    ```python
    text = "Amélioration de l'état général de la patiente"
    doc = nlp(text)
    spans = doc.spans["global_health_status"]

    spans
    # Out: [Amélioration de l'état général]

    span = spans[0]
    span._.global_health_status
    # Out: 'other'
    ```
