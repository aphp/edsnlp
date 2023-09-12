=== "1"
    ```python
    text = "Une fibrose interstitielle diffuse idiopathique"
    doc = nlp(text)
    spans = doc.spans["copd"]

    spans
    # Out: [fibrose interstitielle diffuse idiopathique]
    ```



=== "2"
    ```python
    text = "Patient atteint de pneumoconiose"
    doc = nlp(text)
    spans = doc.spans["copd"]

    spans
    # Out: [pneumoconiose]
    ```



=== "3"
    ```python
    text = "Présence d'une HTAP."
    doc = nlp(text)
    spans = doc.spans["copd"]

    spans
    # Out: [HTAP]
    ```



=== "4"
    ```python
    text = "On voit une hypertension pulmonaire minime"
    doc = nlp(text)
    spans = doc.spans["copd"]

    spans
    # Out: []
    ```



=== "5"
    ```python
    text = "La patiente a été mis sous oxygénorequérance"
    doc = nlp(text)
    spans = doc.spans["copd"]

    spans
    # Out: []
    ```



=== "6"
    ```python
    text = "La patiente est sous oxygénorequérance au long cours"
    doc = nlp(text)
    spans = doc.spans["copd"]

    spans
    # Out: [oxygénorequérance au long cours]

    span = spans[0]

    span._.assigned
    # Out: {'long': [long cours]}
    ```
