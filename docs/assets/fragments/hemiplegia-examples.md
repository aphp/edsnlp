=== "1"
    ```python
    text = "Patient hémiplégique"
    doc = nlp(text)
    spans = doc.spans["hemiplegia"]

    spans
    # Out: [hémiplégique]
    ```



=== "2"
    ```python
    text = "Paralysie des membres inférieurs"
    doc = nlp(text)
    spans = doc.spans["hemiplegia"]

    spans
    # Out: [Paralysie des membres]
    ```



=== "3"
    ```python
    text = "Patient en LIS"
    doc = nlp(text)
    spans = doc.spans["hemiplegia"]

    spans
    # Out: [LIS]
    ```
