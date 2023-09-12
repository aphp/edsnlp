
=== "1"
    ```python
    text = "Présence d'un oedème pulmonaire"
    doc = nlp(text)
    spans = doc.spans["congestive_heart_failure"]

    spans
    # Out: [oedème pulmonaire]
    ```

=== "2"
    ```python
    text = "Le patient est équipé d'un pace-maker"
    doc = nlp(text)
    spans = doc.spans["congestive_heart_failure"]

    spans
    # Out: [pace-maker]
    ```

=== "3"
    ```python
    text = "Un cardiopathie non décompensée"
    doc = nlp(text)
    spans = doc.spans["congestive_heart_failure"]

    spans
    # Out: []
    ```

=== "4"
    ```python
    text = "Insuffisance cardiaque"
    doc = nlp(text)
    spans = doc.spans["congestive_heart_failure"]

    spans
    # Out: [Insuffisance cardiaque]
    ```

=== "5"
    ```python
    text = "Insuffisance cardiaque minime"
    doc = nlp(text)
    spans = doc.spans["congestive_heart_failure"]

    spans
    # Out: []
    ```
