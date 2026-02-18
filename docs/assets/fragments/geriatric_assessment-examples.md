=== "1"
    ```python
    text = "Planification d'évaluation gériatrique."
    doc = nlp(text)
    spans = doc.spans["geriatric_assessment"]

    spans
    # Out: ['évaluation gériatrique']

    span = spans[0]
    span._.geriatric_assessment
    # Out: 'other'
    ```
