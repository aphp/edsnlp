=== "1"
    ```python
    text = "Patient atteint d'une glomérulopathie."
    doc = nlp(text)
    spans = doc.spans["ckd"]

    spans
    # Out: [glomérulopathie]
    ```


=== "2"
    ```python
    text = "Patient atteint d'une tubulopathie aigüe."
    doc = nlp(text)
    spans = doc.spans["ckd"]

    spans
    # Out: []
    ```


=== "3"
    ```python
    text = "Patient transplanté rénal"
    doc = nlp(text)
    spans = doc.spans["ckd"]

    spans
    # Out: [transplanté rénal]
    ```


=== "4"
    ```python
    text = "Présence d'une insuffisance rénale aigüe sur chronique"
    doc = nlp(text)
    spans = doc.spans["ckd"]

    spans
    # Out: [insuffisance rénale aigüe sur chronique]
    ```


=== "5"
    ```python
    text = "Le patient a été dialysé"
    doc = nlp(text)
    spans = doc.spans["ckd"]

    spans
    # Out: []
    ```


=== "6"
    ```python
    text = "Le patient est dialysé chaque lundi"
    doc = nlp(text)
    spans = doc.spans["ckd"]

    spans
    # Out: [dialysé chaque lundi]

    span = spans[0]

    span._.assigned
    # Out: {'chronic': [lundi]}
    ```


=== "7"
    ```python
    text = "Présence d'une IRC"
    doc = nlp(text)
    spans = doc.spans["ckd"]

    spans
    # Out: []
    ```


=== "8"
    ```python
    text = "Présence d'une IRC sévère"
    doc = nlp(text)
    spans = doc.spans["ckd"]

    spans
    # Out: [IRC sévère]

    span = spans[0]

    span._.assigned
    # Out: {'status': sévère}
    ```


=== "9"
    ```python
    text = "Présence d'une IRC au stade IV"
    doc = nlp(text)
    spans = doc.spans["ckd"]

    spans
    # Out: [IRC au stade IV]

    span = spans[0]

    span._.assigned
    # Out: {'stage': IV}
    ```


=== "10"
    ```python
    text = "Présence d'une IRC avec DFG à 30"
    doc = nlp(text)
    spans = doc.spans["ckd"]

    spans
    # Out: [IRC avec DFG à 30]

    span = spans[0]

    span._.assigned
    # Out: {'dfg': 30}
    ```


=== "11"
    ```python
    text = "Présence d'une maladie rénale avec DFG à 110"
    doc = nlp(text)
    spans = doc.spans["ckd"]

    spans
    # Out: []
    ```
