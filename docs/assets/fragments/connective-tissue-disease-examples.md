=== "1"
    ```python
    text = "Présence d'une sclérodermie."
    doc = nlp(text)
    spans = doc.spans["connective_tissue_disease"]

    spans
    # Out: [sclérodermie]
    ```



=== "2"
    ```python
    text = "Patient atteint d'un lupus."
    doc = nlp(text)
    spans = doc.spans["connective_tissue_disease"]

    spans
    # Out: [lupus]
    ```



=== "3"
    ```python
    text = "Présence d'anticoagulants lupiques,"
    doc = nlp(text)
    spans = doc.spans["connective_tissue_disease"]

    spans
    # Out: []
    ```



=== "4"
    ```python
    text = "Il y a une MICI."
    doc = nlp(text)
    spans = doc.spans["connective_tissue_disease"]

    spans
    # Out: [MICI]
    ```



=== "5"
    ```python
    text = "Syndrome de Raynaud"
    doc = nlp(text)
    spans = doc.spans["connective_tissue_disease"]

    spans
    # Out: [Raynaud]
    ```
