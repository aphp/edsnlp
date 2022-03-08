# Hypothesis

The `eds.hypothesis` pipeline uses a simple rule-based algorithm to detect spans that are speculations rather than certain statements.

## Usage

The following snippet matches a simple terminology, and checks whether the extracted entities are part of a speculation. It is complete and can be run _as is_.

```python
import spacy

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
# Out: [douleur, fracture]

doc.ents[0]._.hypothesis
# Out: False

doc.ents[1]._.hypothesis
# Out: True
```

## Configuration

The pipeline can be configured using the following parameters :

| Parameter      | Explanation                                                | Default                           |
| -------------- | ---------------------------------------------------------- | --------------------------------- |
| `attr`         | spaCy attribute to match on (eg `NORM`, `TEXT`, `LOWER`)   | `"NORM"`                          |
| `pseudo`       | Pseudo-hypothesis patterns                                 | `None` (use pre-defined patterns) |
| `preceding`    | Preceding hypothesis patterns                              | `None` (use pre-defined patterns) |
| `following`    | Following hypothesis patterns                              | `None` (use pre-defined patterns) |
| `termination`  | Termination patterns (for syntagma/proposition extraction) | `None` (use pre-defined patterns) |
| `verbs_hyp`    | Patterns for verbs that imply a hypothesis                 | `None` (use pre-defined patterns) |
| `verbs_eds`    | Common verb patterns, checked for conditional mode         | `None` (use pre-defined patterns) |
| `on_ents_only` | Whether to qualify pre-extracted entities only             | `True`                            |
| `within_ents`  | Whether to look for hypothesis within entities             | `False`                           |
| `explain`      | Whether to keep track of the cues for each entity          | `False`                           |

## Declared extensions

The `eds.hypothesis` pipeline declares two [spaCy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects :

1. The `hypothesis` attribute is a boolean, set to `True` if the pipeline predicts that the span/token is a speculation.
2. The `hypothesis_` property is a human-readable string, computed from the `hypothesis` attribute. It implements a simple getter function that outputs `HYP` or `CERT`, depending on the value of `hypothesis`.

## Performance

The pipeline's performance is measured on three datasets :

- The ESSAI[@dalloux2017ESSAI] and CAS[@grabar2018CAS] datasets were developed at the CNRS. The two are concatenated.
- The NegParHyp corpus was specifically developed at EDS to test the pipeline on actual clinical notes, using pseudonymised notes from the EDS.

| Dataset   | Hypothesis F1 |
| --------- | ------------- |
| CAS/ESSAI | 49%           |
| NegParHyp | 52%           |

!!! note "NegParHyp corpus"

    The NegParHyp corpus was built by matching a subset of the MeSH terminology with around 300 documents
    from AP-HP's clinical data warehouse.
    Matched entities were then labelled for negation, speculation and family context.

## Authors and citation

The `eds.hypothesis` pipeline was developed by AP-HP's Data Science team.

\bibliography
