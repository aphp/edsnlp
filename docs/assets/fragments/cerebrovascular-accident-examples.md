=== "1"
    ```python
    text = "Patient hospitalisé à AVC."
    doc = nlp(text)
    spans = doc.spans["cerebrovascular_accident"]

    spans
    # Out: []
    ```



=== "2"
    ```python
    text = "Hospitalisation pour un AVC."
    doc = nlp(text)
    spans = doc.spans["cerebrovascular_accident"]

    spans
    # Out: [AVC]
    ```



=== "3"
    ```python
    text = "Saignement intracranien"
    doc = nlp(text)
    spans = doc.spans["cerebrovascular_accident"]

    spans
    # Out: [Saignement]

    span = spans[0]

    span._.assigned
    # Out: {'brain_localized': [intracranien]}
    ```



=== "4"
    ```python
    text = "Thrombose périphérique"
    doc = nlp(text)
    spans = doc.spans["cerebrovascular_accident"]

    spans
    # Out: []
    ```



=== "5"
    ```python
    text = "Thrombose sylvienne"
    doc = nlp(text)
    spans = doc.spans["cerebrovascular_accident"]

    spans
    # Out: [Thrombose]

    span = spans[0]

    span._.assigned
    # Out: {'brain_localized': [sylvienne]}
    ```



=== "6"
    ```python
    text = "Infarctus cérébral"
    doc = nlp(text)
    spans = doc.spans["cerebrovascular_accident"]

    spans
    # Out: [Infarctus]

    span = spans[0]

    span._.assigned
    # Out: {'brain_localized': [cérébral]}
    ```



=== "7"
    ```python
    text = "Soigné via un thrombolyse"
    doc = nlp(text)
    spans = doc.spans["cerebrovascular_accident"]

    spans
    # Out: [thrombolyse]
    ```
