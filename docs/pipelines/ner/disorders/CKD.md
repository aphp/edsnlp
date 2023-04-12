# CKD

The `eds.CKD` pipeline component extracts mentions of CKD (Chronic Kidney Disease). It will notably match:

- Mentions of various diseases (see below)
- Kidney transplantation
- Chronic dialysis
- Renal failure **from stage 3 to 5**. The stage is extracted by trying 3 methods:
    - Extracting the mentionned stage directly ("*IRC stade IV*")
    - Extracting the severity directly ("*IRC terminale*")
    - Extracting the mentionned GFR (DFG in french) ("*IRC avec DFG estimé à 30 mL/min/1,73m2)*")

??? info "Details of the used patterns"
    <!-- no-check -->
    ```python
    # fmt: off
    --8<-- "edsnlp/pipelines/ner/disorders/CKD/patterns.py"
    # fmt: on
    ```

## Extensions

On each span `span` that match, the following attributes are available:

- `span._.status_`: set to `"PRESENT"`
- `span._.assigned`: dictionary with the following keys, if relevant:
    - `stage`: mentionned renal failure stage
    - `status`: mentionned renal failure severity (e.g. modérée, sévère, terminale, etc.)
    - `dfg`: mentionned DFG

## Usage


```python
import spacy

nlp = spacy.blank("eds")
nlp.add_pipe("eds.sentences")
nlp.add_pipe(
    "eds.normalizer",
    config=dict(
        accents=True,
        lowercase=True,
        quotes=True,
        spaces=True,
        pollution=dict(
            information=True,
            bars=True,
            biology=True,
            doctors=True,
            web=True,
            coding=True,
            footer=True,
        ),
    ),
)
nlp.add_pipe(f"eds.CKD")
```

Below are a few examples:




=== "1"
    ```python
    text = "Patient atteint d'une glomérulopathie."
    doc = nlp(text)
    spans = doc.spans["CKD"]

    spans
    # Out: [glomérulopathie]
    ```



=== "2"
    ```python
    text = "Patient atteint d'une tubulopathie aigüe."
    doc = nlp(text)
    spans = doc.spans["CKD"]

    spans
    # Out: []
    ```



=== "3"
    ```python
    text = "Patient transplanté rénal"
    doc = nlp(text)
    spans = doc.spans["CKD"]

    spans
    # Out: [transplanté rénal]
    ```



=== "4"
    ```python
    text = "Présence d'une insuffisance rénale aigüe sur chronique"
    doc = nlp(text)
    spans = doc.spans["CKD"]

    spans
    # Out: [insuffisance rénale aigüe sur chronique]
    ```



=== "5"
    ```python
    text = "Le patient a été dialysé"
    doc = nlp(text)
    spans = doc.spans["CKD"]

    spans
    # Out: []
    ```



=== "6"
    ```python
    text = "Le patient est dialysé chaque lundi"
    doc = nlp(text)
    spans = doc.spans["CKD"]

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
    spans = doc.spans["CKD"]

    spans
    # Out: []
    ```



=== "8"
    ```python
    text = "Présence d'une IRC sévère"
    doc = nlp(text)
    spans = doc.spans["CKD"]

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
    spans = doc.spans["CKD"]

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
    spans = doc.spans["CKD"]

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
    spans = doc.spans["CKD"]

    spans
    # Out: []
    ```

## Authors and citation

The `eds.CKD` component was developed by AP-HP's Data Science team with a team of medical experts. A paper describing in details the development of those components is being drafted and will soon be available.
