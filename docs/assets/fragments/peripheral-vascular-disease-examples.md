=== "1"
    ```python
    text = "Un AOMI"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: [AOMI]
    ```



=== "2"
    ```python
    text = "Présence d'un infarctus rénal"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: [infarctus rénal]
    ```



=== "3"
    ```python
    text = "Une angiopathie cérébrale"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: []
    ```



=== "4"
    ```python
    text = "Une angiopathie"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: [angiopathie]
    ```



=== "5"
    ```python
    text = "Une thrombose cérébrale"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: []
    ```



=== "6"
    ```python
    text = "Une thrombose des veines superficielles"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: []
    ```



=== "7"
    ```python
    text = "Une thrombose"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: [thrombose]
    ```



=== "8"
    ```python
    text = "Effectuer un bilan pre-trombose"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: []
    ```



=== "9"
    ```python
    text = "Une ischémie des MI est remarquée."
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: [ischémie des MI]

    span = spans[0]

    span._.assigned
    # Out: {'peripheral': [MI]}
    ```



=== "10"
    ```python
    text = "Plusieurs cas d'EP"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: [EP]
    ```



=== "11"
    ```python
    text = "Effectuer des cures d'EP"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: []
    ```



=== "12"
    ```python
    text = "Le patient est hypertendu"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: [hypertendu]
    ```



=== "13"
    ```python
    text = "Une hypertension portale"
    doc = nlp(text)
    spans = doc.spans["peripheral_vascular_disease"]

    spans
    # Out: []
    ```
