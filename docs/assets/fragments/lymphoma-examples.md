=== "1"
    ```python
    text = "Un lymphome de Hodgkin."
    doc = nlp(text)
    spans = doc.spans["lymphoma"]

    spans
    # Out: [lymphome de Hodgkin]
    ```



=== "2"
    ```python
    text = "Atteint d'un Waldenstörm"
    doc = nlp(text)
    spans = doc.spans["lymphoma"]

    spans
    # Out: [Waldenstörm]
    ```



=== "3"
    ```python
    text = "Un LAGC"
    doc = nlp(text)
    spans = doc.spans["lymphoma"]

    spans
    # Out: [LAGC]
    ```



=== "4"
    ```python
    text = "anti LAGC: 10^4/mL"
    doc = nlp(text)
    spans = doc.spans["lymphoma"]

    spans
    # Out: []
    ```
