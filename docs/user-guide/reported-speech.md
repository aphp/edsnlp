# Reported Speech

The `rspeech` pipeline uses a simple rule-based algorithm to detect spans that relate to reported speech (eg when the doctor quotes the patient). It was designed at AP-HP's EDS.

## Scope

The `rspeech` pipeline can functions in two modes :

1. Annotation of the extracted entities (this is the default). To increase throughput, only preextracted entities (found in `doc.ents`) are processed and tagged as positive or negative.
2. Full-text, token-wise annotation. This mode is activated with by setting the `on_ents_only` parameter to `False`.

Since the natural way to use EDS-NLP is to extract entities and then check their polarity, the second mode is generally unused.

## Declared extensions

The `rspeech` pipeline declares two [Spacy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects :

1. The `reported_speech` attribute is a boolean, set to `True` if the pipeline predicts that the span/token is reported.
2. The `reported_speech_` property is a human-readable string, computed from the `reported_speech` attribute. It implements a simple getter function that outputs `DIRECT` or `REPORTED`, depending on the value of `reported_speech`.

## Usage

The following snippet matches a simple terminology, and checks the polarity of the extracted entities.

```python
import spacy
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("sentences")
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

doc.ents[0]._.polarity_
# Out: 'DIRECT'

doc.ents[1]._.polarity_
# Out: 'REPORTED'
```

## Performance

The pipeline's performance is still being evaluated.

## Authors and citation

The `rspeech` pipeline was developed by the Data Science team at EDS.
