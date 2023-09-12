=== "SIDA"
    ```python
    text = "Patient atteint du VIH au stade SIDA."
    doc = nlp(text)
    spans = doc.spans["aids"]

    spans
    # Out: [VIH au stade SIDA]
    ```



=== "VIH"
    ```python
    text = "Patient atteint du VIH."
    doc = nlp(text)
    spans = doc.spans["aids"]

    spans
    # Out: []
    ```



=== "Coinfection"
    ```python
    text = "Il y a un VIH avec coinfection pneumocystose"
    doc = nlp(text)
    spans = doc.spans["aids"]

    spans
    # Out: [VIH]

    span = spans[0]

    span._.assigned
    # Out: {'opportunist': [coinfection, pneumocystose]}
    ```



=== "VIH stade SIDA"
    ```python
    text = "Présence d'un VIH stade C"
    doc = nlp(text)
    spans = doc.spans["aids"]

    spans
    # Out: [VIH]

    span = spans[0]

    span._.assigned
    # Out: {'stage': [C]}
    ```
