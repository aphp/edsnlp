# Hypothesis

The `eds.hypothesis` pipeline uses a simple rule-based algorithm to detect spans that are speculations rather than certain statements. It was designed at AP-HP's EDS.

## Declared extensions

The `eds.hypothesis` pipeline declares two [Spacy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects :

1. The `hypothesis` attribute is a boolean, set to `True` if the pipeline predicts that the span/token is a speculation.
2. The `hypothesis_` property is a human-readable string, computed from the `hypothesis` attribute. It implements a simple getter function that outputs `HYP` or `CERT`, depending on the value of `hypothesis`.

## Configuration

The pipeline can be configured using the following parameters :

| Parameter      | Explanation                                                | Default                           |
| -------------- | ---------------------------------------------------------- | --------------------------------- |
| `attr`         | Spacy attribute to match on (eg `NORM`, `TEXT`, `LOWER`)   | `"NORM"`                          |
| `pseudo`       | Pseudo-hypothesis patterns                                 | `None` (use pre-defined patterns) |
| `preceding`    | Preceding hypothesis patterns                              | `None` (use pre-defined patterns) |
| `following`    | Following hypothesis patterns                              | `None` (use pre-defined patterns) |
| `termination`  | Termination patterns (for syntagma/proposition extraction) | `None` (use pre-defined patterns) |
| `verbs_hyp`    | Patterns for verbs that imply a hypothesis                 | `None` (use pre-defined patterns) |
| `verbs_eds`    | Common verb patterns, checked for conditional mode         | `None` (use pre-defined patterns) |
| `on_ents_only` | Whether to qualify pre-extracted entities only             | `True`                            |
| `within_ents`  | Whether to look for hypothesis within entities             | `False`                           |
| `explain`      | Whether to keep track of the cues for each entity          | `False`                           |

## Usage

The following snippet matches a simple terminology, and checks the family context of the extracted entities. It is complete and can be run _as is_.

```python
import spacy
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("eds.sentences")
# Dummy matcher
nlp.add_pipe(
    "eds.matcher",
    config=dict(terms=dict(douleur="douleur", fracture="fracture")),
)
nlp.add_pipe("eds.hypothesis")

text = (
    "Le patient est admis le 23 août 2021 pour une douleur au bras. "
    "Possible fracture du radius."
)

doc = nlp(text)

doc.ents
# Out: [patient, fracture]

doc.ents[0]._.hypothesis
# Out: False

doc.ents[1]._.hypothesis
# Out: True
```

## Performance

The pipeline's performance is measured on three datasets :

- The ESSAI ({footcite:t}`dalloux:hal-01659637`) and CAS ({footcite:t}`grabar:hal-01937096`) datasets were developped at the CNRS. The two are concatenated.
- The NegParHyp corpus was specifically developed at EDS to test the pipeline on actual medical notes, using pseudonymised notes from the EDS.

| Version | Dataset   | Hypothesis F1 |
| ------- | --------- | ------------- |
| v0.0.1  | CAS/ESSAI | 48%           |
| v0.0.2  | CAS/ESSAI | 49%           |
| v0.0.2  | NegParHyp | 52%           |

## Authors and citation

The `eds.hypothesis` pipeline was developed at the Data and Innovation unit, IT department, AP-HP.

## References

```{eval-rst}
.. footbibliography::
```
