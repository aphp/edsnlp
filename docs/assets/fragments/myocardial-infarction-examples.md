=== "1"
    ```python
    text = "Une cardiopathie ischémique"
    doc = nlp(text)
    spans = doc.spans["myocardial_infarction"]

    spans
    # Out: [cardiopathie ischémique]
    ```



=== "2"
    ```python
    text = "Une cardiopathie non-ischémique"
    doc = nlp(text)
    spans = doc.spans["myocardial_infarction"]

    spans
    # Out: []
    ```



=== "3"
    ```python
    text = "Présence d'un stent sur la marginale"
    doc = nlp(text)
    spans = doc.spans["myocardial_infarction"]

    spans
    # Out: [stent sur la marginale]

    span = spans[0]

    span._.assigned
    # Out: {'heart_localized': [marginale]}
    ```



=== "4"
    ```python
    text = "Présence d'un stent périphérique"
    doc = nlp(text)
    spans = doc.spans["myocardial_infarction"]

    spans
    # Out: []
    ```



=== "5"
    ```python
    text = "infarctus du myocarde"
    doc = nlp(text)
    spans = doc.spans["myocardial_infarction"]

    spans
    # Out: [infarctus du myocarde]

    span = spans[0]

    span._.assigned
    # Out: {'heart_localized': [myocarde]}
    ```
