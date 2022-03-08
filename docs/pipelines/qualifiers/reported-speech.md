# Reported Speech

The `eds.reported_speech` pipeline uses a simple rule-based algorithm to detect spans that relate to reported speech (eg when the doctor quotes the patient). It was designed at AP-HP's EDS.

## Usage

The following snippet matches a simple terminology, and checks the polarity of the extracted entities. It is complete and can be run _as is_.

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.sentences")
# Dummy matcher
nlp.add_pipe(
    "eds.matcher",
    config=dict(terms=dict(patient="patient", alcool="alcoolisé")),
)
nlp.add_pipe("eds.reported_speech")

text = (
    "Le patient est admis aux urgences ce soir pour une douleur au bras. "
    "Il nie être alcoolisé."
)

doc = nlp(text)

doc.ents
# Out: [patient, alcoolisé]

doc.ents[0]._.reported_speech
# Out: False

doc.ents[1]._.reported_speech
# Out: True
```

## Configuration

The pipeline can be configured using the following parameters :

| Parameter      | Explanation                                                | Default                           |
| -------------- | ---------------------------------------------------------- | --------------------------------- |
| `attr`         | spaCy attribute to match on (eg `NORM`, `TEXT`, `LOWER`)   | `"NORM"`                          |
| `pseudo`       | Pseudo-reported speed patterns                             | `None` (use pre-defined patterns) |
| `preceding`    | Preceding reported speed patterns                          | `None` (use pre-defined patterns) |
| `following`    | Following reported speed patterns                          | `None` (use pre-defined patterns) |
| `termination`  | Termination patterns (for syntagma/proposition extraction) | `None` (use pre-defined patterns) |
| `verbs`        | Patterns for verbs that imply a reported speed             | `None` (use pre-defined patterns) |
| `on_ents_only` | Whether to qualify pre-extracted entities only             | `True`                            |
| `within_ents`  | Whether to look for reported speed within entities         | `False`                           |
| `explain`      | Whether to keep track of the cues for each entity          | `False`                           |

## Declared extensions

The `eds.reported_speech` pipeline declares two [spaCy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects :

1. The `reported_speech` attribute is a boolean, set to `True` if the pipeline predicts that the span/token is reported.
2. The `reported_speech_` property is a human-readable string, computed from the `reported_speech` attribute. It implements a simple getter function that outputs `DIRECT` or `REPORTED`, depending on the value of `reported_speech`.

## Authors and citation

The `eds.reported_speech` pipeline was developed by AP-HP's Data Science team.
