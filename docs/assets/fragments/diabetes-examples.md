=== "1"
    ```python
    text = "Présence d'un DT2"
    doc = nlp(text)
    spans = doc.spans["diabetes"]

    spans
    # Out: [DT2]
    ```



=== "2"
    ```python
    text = "Présence d'un DNID"
    doc = nlp(text)
    spans = doc.spans["diabetes"]

    spans
    # Out: [DNID]
    ```



=== "3"
    ```python
    text = "Patient diabétique"
    doc = nlp(text)
    spans = doc.spans["diabetes"]

    spans
    # Out: [diabétique]
    ```



=== "4"
    ```python
    text = "Un diabète insipide"
    doc = nlp(text)
    spans = doc.spans["diabetes"]

    spans
    # Out: []
    ```



=== "5"
    ```python
    text = "Atteinte neurologique d'origine diabétique"
    doc = nlp(text)
    spans = doc.spans["diabetes"]

    spans
    # Out: [origine diabétique]

    span = spans[0]

    span._.detailed_status
    # Out: WITH_COMPLICATION

    span._.assigned
    # Out: {'complicated_before': [origine]}
    ```



=== "6"
    ```python
    text = "Une rétinopathie diabétique"
    doc = nlp(text)
    spans = doc.spans["diabetes"]

    spans
    # Out: [rétinopathie diabétique]

    span = spans[0]

    span._.detailed_status
    # Out: WITH_COMPLICATION

    span._.assigned
    # Out: {'complicated_before': [rétinopathie]}
    ```



=== "7"
    ```python
    text = "Il y a un mal perforant plantaire"
    doc = nlp(text)
    spans = doc.spans["diabetes"]

    spans
    # Out: [mal perforant plantaire]

    span = spans[0]

    span._.detailed_status
    # Out: WITH_COMPLICATION
    ```
