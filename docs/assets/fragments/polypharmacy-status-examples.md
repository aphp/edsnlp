=== "1"
    ```python
    text = "On constate une iatrogénie"
    doc = nlp(text)
    spans = doc.spans["polypharmacy_status"]

    spans
    # Out: [iatrogénie]

    span = spans[0]
    span._.polypharmacy_status
    # Out: 'altered_nondescript'
    ```
