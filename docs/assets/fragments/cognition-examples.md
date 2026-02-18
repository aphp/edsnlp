=== "1"
    ```python
    text = "Le patient a un langage fluent et cohérent"
    doc = nlp(text)
    spans = doc.spans["cognition"]

    spans
    # Out: ['langage fluent et cohérent']

    span = spans[0]
    span._.cognition
    # Out: 'healthy'
    ```



=== "2"
    ```python
    text = "Période de confusion aiguë"
    doc = nlp(text)
    spans = doc.spans["cognition"]

    spans
    # Out: []
    ```



=== "3"
    ```python
    text = "Signes de démence"
    doc = nlp(text)
    spans = doc.spans["cognition"]

    spans
    # Out: ['démence']

    span = spans[0]
    span._.cognition
    # Out: 'altered_severe'
    ```
