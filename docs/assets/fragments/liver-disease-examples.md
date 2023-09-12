=== "1"
    ```python
    text = "Il y a une fibrose hépatique"
    doc = nlp(text)
    spans = doc.spans["liver_disease"]

    spans
    # Out: [fibrose hépatique]
    ```



=== "2"
    ```python
    text = "Une hépatite B chronique"
    doc = nlp(text)
    spans = doc.spans["liver_disease"]

    spans
    # Out: [hépatite B chronique]
    ```



=== "3"
    ```python
    text = "Le patient consulte pour une cirrhose"
    doc = nlp(text)
    spans = doc.spans["liver_disease"]

    spans
    # Out: [cirrhose]

    span = spans[0]

    span._.detailed_status
    # Out: MODERATE_TO_SEVERE
    ```



=== "4"
    ```python
    text = "Greffe hépatique."
    doc = nlp(text)
    spans = doc.spans["liver_disease"]

    spans
    # Out: [Greffe hépatique]

    span = spans[0]

    span._.detailed_status
    # Out: MODERATE_TO_SEVERE
    ```
