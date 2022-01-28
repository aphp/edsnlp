# Negation

The `negation` pipeline uses a simple rule-based algorithm to detect negated spans. It was designed at AP-HP's EDS, following the insights of the NegEx algorithm by {footcite:t}`chapman_simple_2001`.

## Declared extensions

The `negation` pipeline declares two [Spacy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects :

1. The `negated` attribute is a boolean, set to `True` if the pipeline predicts that the span/token is negated.
2. The `polarity_` property is a human-readable string, computed from the `negated` attribute. It implements a simple getter function that outputs `AFF` or `NEG`, depending on the value of `negated`.

## Rationale

The `negation` pipeline is a rule-based algorithm for detecting negated entities. It functions as follows :

1. The pipeline extracts negation cues. We define three (overlapping) kinds :

   - `preceding`, ie cues that _precede_ negated entities ;
   - `following`, ie cues that _follow_ negated entities ;
   - `verbs`, ie verbs that convey a negation (treated as preceding negations).
     The pipeline also detects _pseudo-negations_, eg `sans doute`/`without doubt` : phrases that contain negation cues, but that are not cues themselves.

2. The pipeline splits the text between sentences and propositions, using annotations from a sentencizer pipeline and `termination` patterns, which define syntagma/proposition terminations.

3. For each pre-extracted entity, the pipeline checks whether there is a cue between the start of the syntagma and the start of the entity, or a following cue between the end of the entity and the end of the proposition.

Albeit simple, the `negation` pipeline achieves 88% F1-score on our dataset.

## Configuration

The pipeline can be configured using the following parameters :

| Parameter      | Explanation                                                | Default                           |
| -------------- | ---------------------------------------------------------- | --------------------------------- |
| `attr`         | Spacy attribute to match on (eg `NORM`, `TEXT`, `LOWER`)   | `"NORM"`                          |
| `pseudo`       | Pseudo-negation patterns                                   | `None` (use pre-defined patterns) |
| `preceding`    | Preceding negation patterns                                | `None` (use pre-defined patterns) |
| `following`    | Following negation patterns                                | `None` (use pre-defined patterns) |
| `termination`  | Termination patterns (for syntagma/proposition extraction) | `None` (use pre-defined patterns) |
| `verbs`        | Patterns for verbs that imply a negation                   | `None` (use pre-defined patterns) |
| `on_ents_only` | Whether to qualify pre-extracted entities only             | `True`                            |
| `within_ents`  | Whether to look for negations within entities              | `False`                           |
| `explain`      | Whether to keep track of the cues for each entity          | `False`                           |

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
    config=dict(terms=dict(patient="patient", fracture="fracture")),
)
nlp.add_pipe("negation")

text = (
    "Le patient est admis le 23 août 2021 pour une douleur au bras. "
    "Le scanner ne détecte aucune fracture."
)

doc = nlp(text)

doc.ents
# Out: [patient, fracture]

doc.ents[0]._.polarity_
# Out: 'AFF'

doc.ents[1]._.polarity_
# Out: 'NEG'
```

## Performance

The pipeline's performance is measured on three datasets :

- The ESSAI ({footcite:t}`dalloux:hal-01659637`) and CAS ({footcite:t}`grabar:hal-01937096`) datasets were developped at the CNRS. The two are concatenated.
- The NegParHyp corpus was specifically developed at EDS to test the pipeline on actual medical notes, using pseudonymised notes from the EDS.

| Version | Dataset   | Negation F1 |
| ------- | --------- | ----------- |
| v0.0.1  | CAS/ESSAI | 79%         |
| v0.0.2  | CAS/ESSAI | 71%         |
| v0.0.2  | NegParHyp | 88%         |

Note that we favour the NegParHyp corpus, since it is comprised of actual medical notes from the data warehouse. The table shows that the pipeline does not perform as well on other datasets.

## Authors and citation

The `negation` pipeline was developed at the Data and Innovation unit, IT department, AP-HP.

## References

```{eval-rst}
.. footbibliography::
```
