=== "1"
    ```python
    text = "Beaucoup d'ulcères gastriques"
    doc = nlp(text)
    spans = doc.spans["peptic_ulcer_disease"]

    spans
    # Out: [ulcères gastriques]
    ```



=== "2"
    ```python
    text = "Présence d'UGD"
    doc = nlp(text)
    spans = doc.spans["peptic_ulcer_disease"]

    spans
    # Out: [UGD]
    ```



=== "3"
    ```python
    text = "La patient à des ulcères"
    doc = nlp(text)
    spans = doc.spans["peptic_ulcer_disease"]

    spans
    # Out: []
    ```



=== "4"
    ```python
    text = "Au niveau gastrique: blabla blabla blabla blabla blabla quelques ulcères"
    doc = nlp(text)
    spans = doc.spans["peptic_ulcer_disease"]

    spans
    # Out: [gastrique: blabla blabla blabla blabla blabla quelques ulcères]

    span = spans[0]

    span._.assigned
    # Out: {'is_peptic': [gastrique]}
    ```
