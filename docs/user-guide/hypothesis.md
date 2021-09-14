# Hypothesis

The `hypothesis` pipeline uses a simple rule-based algorithm to detect spans that are speculations rather than certain statements. It was designed at AP-HP's EDS.

## Scope

The `hypothesis` pipeline can functions in two modes :

1. Annotation of the extracted entities (this is the default). To increase throughput, only pre-extracted entities (found in `doc.ents`) are processed.
2. Full-text, token-wise annotation. This mode is activated with by setting the `on_ents_only` parameter to `False`.

Since the natural way to use EDS-NLP is to extract entities and then check whether they are related to the patient themself, the second mode is generally unused.

## Declared extensions

The `hypothesis` pipeline declares two [Spacy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects :

1. The `hypothesis` attribute is a boolean, set to `True` if the pipeline predicts that the span/token is a speculation.
2. The `hypothesis_` property is a human-readable string, computed from the `hypothesis` attribute. It implements a simple getter function that outputs `HYP` or `CERT`, depending on the value of `hypothesis`.

## Usage

The following snippet matches a simple terminology, and checks the family context of the extracted entities. It is complete, and should run _as is_.

```python
import spacy
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("sentences")
nlp.add_pipe(
    "matcher",
    config=dict(terms=dict(douleur="douleur", fracture="fracture")),
)
nlp.add_pipe("hypothesis")

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

The pipeline's performance are measured on three datasets :

- The ESSAI ({footcite:t}`dalloux:hal-01659637`) and CAS ({footcite:t}`grabar:hal-01937096`) datasets were developped at the CNRS. See the [dedicated EDS-Datasets page](https://equipedatascience-pages.eds.aphp.fr/eds-datasets/datasets/essai-cas.html) for more information.
- The NegParHyp corpus was specifically developed at EDS to test the pipeline on actual medical notes, using pseudonymised notes from the EDS. See [EDS-Datasets](https://equipedatascience-pages.eds.aphp.fr/eds-datasets/datasets/negparhyp.html) for more information and access to the annotation guide.

| Version | Dataset   | Hypothesis F1 |
| ------- | --------- | ------------- |
| v0.0.1  | CAS/ESSAI | 48%           |
| v0.0.2  | CAS/ESSAI | 49%           |
| v0.0.2  | NegParHyp | 52%           |

## Authors and citation

The `family` pipeline was developed by the Data Science team at EDS.

## References

```{eval-rst}
.. footbibliography::
```
