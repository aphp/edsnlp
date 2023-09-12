=== "1"
    ```python
    text = "Présence d'un carcinome intra-hépatique."
    doc = nlp(text)
    spans = doc.spans["solid_tumor"]

    spans
    # Out: [carcinome]
    ```



=== "2"
    ```python
    text = "Patient avec un K sein."
    doc = nlp(text)
    spans = doc.spans["solid_tumor"]

    spans
    # Out: [K sein]
    ```



=== "3"
    ```python
    text = "Il y a une tumeur bénigne"
    doc = nlp(text)
    spans = doc.spans["solid_tumor"]

    spans
    # Out: []
    ```



=== "4"
    ```python
    text = "Tumeur métastasée"
    doc = nlp(text)
    spans = doc.spans["solid_tumor"]

    spans
    # Out: [Tumeur métastasée]

    span = spans[0]

    span._.detailed_status
    # Out: METASTASIS

    span._.assigned
    # Out: {'metastasis': métastasée}
    ```



=== "5"
    ```python
    text = "Cancer du poumon au stade 4"
    doc = nlp(text)
    spans = doc.spans["solid_tumor"]

    spans
    # Out: [Cancer du poumon au stade 4]

    span = spans[0]

    span._.detailed_status
    # Out: METASTASIS

    span._.assigned
    # Out: {'stage': 4}
    ```



=== "6"
    ```python
    text = "Cancer du poumon au stade 2"
    doc = nlp(text)
    spans = doc.spans["solid_tumor"]

    spans
    # Out: [Cancer du poumon au stade 2]

    span = spans[0]

    span._.assigned
    # Out: {'stage': 2}
    ```



=== "7"
    ```python
    text = "Présence de nombreuses lésions secondaires"
    doc = nlp(text)
    spans = doc.spans["solid_tumor"]

    spans
    # Out: [lésions secondaires]

    span = spans[0]

    span._.detailed_status
    # Out: METASTASIS
    ```
