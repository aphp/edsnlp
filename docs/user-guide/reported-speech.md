# Reported Speech

The `rspeech` pipeline uses a simple rule-based algorithm to detect spans that relate to reported speech (eg when the doctor quotes the patient). It was designed at AP-HP's EDS.

## Declared extensions

The `rspeech` pipeline declares two [Spacy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects :

1. The `reported_speech` attribute is a boolean, set to `True` if the pipeline predicts that the span/token is reported.
2. The `reported_speech_` property is a human-readable string, computed from the `reported_speech` attribute. It implements a simple getter function that outputs `DIRECT` or `REPORTED`, depending on the value of `reported_speech`.

## Usage

The following snippet matches a simple terminology, and checks the polarity of the extracted entities. It is complete and can be run _as is_.

```python
import spacy
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("sentences")
# Dummy matcher
nlp.add_pipe(
    "matcher",
    config=dict(terms=dict(patient="patient", alcool="alcoolisé")),
)
nlp.add_pipe("rspeech")

text = (
    "Le patient est admis aux urgences ce soir pour une douleur au bras. "
    "Il nie être alcoolisé."
)

doc = nlp(text)

doc.ents
# Out: [patient, alcoolisé]

doc.ents[0]._.reported_speech_
# Out: 'DIRECT'

doc.ents[1]._.reported_speech_
# Out: 'REPORTED'
```

## Performance

The pipeline's performance is still being evaluated.

## Authors and citation

The `rspeech` pipeline was developed at the Data and Innovation unit, IT department, AP-HP.
