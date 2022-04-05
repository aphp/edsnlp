# Negation

The `eds.negation` pipeline uses a simple rule-based algorithm to detect negated spans. It was designed at AP-HP's EDS, following the insights of the NegEx algorithm by Chapman et al[@chapman_simple_2001].

## Usage

The following snippet matches a simple terminology, and checks the polarity of the extracted entities. It is complete and can be run _as is_.

```python
import spacy

nlp = spacy.blank("fr")
nlp.add_pipe("eds.sentences")
# Dummy matcher
nlp.add_pipe(
    "eds.matcher",
    config=dict(terms=dict(patient="patient", fracture="fracture")),
)
nlp.add_pipe("eds.negation")

text = (
    "Le patient est admis le 23 août 2021 pour une douleur au bras. "
    "Le scanner ne détecte aucune fracture."
)

doc = nlp(text)

doc.ents
# Out: (patient, fracture)

doc.ents[0]._.negation  # (1)
# Out: False

doc.ents[1]._.negation
# Out: True
```

1. The result of the pipeline is kept in the `negation` custom extension.

## Configuration

The pipeline can be configured using the following parameters :

| Parameter      | Explanation                                                | Default                           |
| -------------- | ---------------------------------------------------------- | --------------------------------- |
| `attr`         | spaCy attribute to match on (eg `NORM`, `TEXT`, `LOWER`)   | `"NORM"`                          |
| `pseudo`       | Pseudo-negation patterns                                   | `None` (use pre-defined patterns) |
| `preceding`    | Preceding negation patterns                                | `None` (use pre-defined patterns) |
| `following`    | Following negation patterns                                | `None` (use pre-defined patterns) |
| `termination`  | Termination patterns (for syntagma/proposition extraction) | `None` (use pre-defined patterns) |
| `verbs`        | Patterns for verbs that imply a negation                   | `None` (use pre-defined patterns) |
| `on_ents_only` | Whether to qualify pre-extracted entities only             | `True`                            |
| `within_ents`  | Whether to look for negations within entities              | `False`                           |
| `explain`      | Whether to keep track of the cues for each entity          | `False`                           |

## Declared extensions

The `eds.negation` pipeline declares two [spaCy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects :

1. The `negation` attribute is a boolean, set to `True` if the pipeline predicts that the span/token is negated.
2. The `negation_` property is a human-readable string, computed from the `negation` attribute. It implements a simple getter function that outputs `AFF` or `NEG`, depending on the value of `negation`.

## Performance

The pipeline's performance is measured on three datasets :

- The ESSAI[@dalloux2017ESSAI] and CAS[@grabar2018CAS] datasets were developed at the CNRS. The two are concatenated.
- The NegParHyp corpus was specifically developed at AP-HP to test the pipeline on actual clinical notes, using pseudonymised notes from the AP-HP.

| Dataset   | Negation F1 |
| --------- | ----------- |
| CAS/ESSAI | 71%         |
| NegParHyp | 88%         |

!!! note "NegParHyp corpus"

    The NegParHyp corpus was built by matching a subset of the MeSH terminology with around 300 documents
    from AP-HP's clinical data warehouse.
    Matched entities were then labelled for negation, speculation and family context.

## Authors and citation

The `eds.negation` pipeline was developed by AP-HP's Data Science team.

\bibliography
