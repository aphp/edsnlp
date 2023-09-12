=== "1"
    ```python
    text = "Sydrome myéloprolifératif"
    doc = nlp(text)
    spans = doc.spans["leukemia"]

    spans
    # Out: [myéloprolifératif]
    ```



=== "2"
    ```python
    text = "Sydrome myéloprolifératif bénin"
    doc = nlp(text)
    spans = doc.spans["leukemia"]

    spans
    # Out: []
    ```



=== "3"
    ```python
    text = "Patient atteint d'une LAM"
    doc = nlp(text)
    spans = doc.spans["leukemia"]

    spans
    # Out: [LAM]
    ```



=== "4"
    ```python
    text = "Une maladie de Vaquez"
    doc = nlp(text)
    spans = doc.spans["leukemia"]

    spans
    # Out: [Vaquez]
    ```
