# Negation

The `negation` pipeline uses a simple rule-based algorithm to detect negated spans. It was designed at AP-HP's EDS, following the insights of the NegEx algorithm by {footcite:t}`chapman_simple_2001`.

## Scope

The `negation` pipeline can functions in two modes :

1. Annotation of the extracted entities (this is the default). To increase throughput, only preextracted entities (found in `doc.ents`) are processed and tagged as positive or negative.
2. Full-text, token-wise annotation. This mode is activated with by setting the `on_ents_only` parameter to `False`.

Since the natural way to use EDS-NLP is to extract entities and then check their polarity, the second mode is generally unused.

## Declared extensions

The `negation` pipeline declares two [Spacy extensions](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on both `Span` and `Token` objects :

1. The `negated` attribute is a boolean, set to `True` if the pipeline predicts that the span/token is negated.
2. The `polarity_` property is a human-readable string, computed from the `negated` attribute. It implements a simple getter function that outputs `AFF` or `NEG`, depending on the value of `negated`.

## Usage

The following snippet matches a simple terminology, and checks the polarity of the extracted entities.

```python
import spacy
from edsnlp import components

nlp = spacy.blank("fr")
nlp.add_pipe("sentences")
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

The pipeline's performance are measured on three datasets

- The ESSAI ({footcite:t}`dalloux:hal-01659637`) and CAS ({footcite:t}`grabar:hal-01937096`) datasets were developped at the CNRS. See the [dedicated EDS-Datasets page](https://equipedatascience-pages.eds.aphp.fr/eds-datasets/datasets/essai-cas.html) for more information.
- The NegParHyp corpus was specifically developed at EDS to test the pipeline on actual medical notes, using pseudonymised notes from the EDS. See [EDS-Datasets](https://equipedatascience-pages.eds.aphp.fr/eds-datasets/datasets/negparhyp.html) for more information.

| Version | Dataset   | Negation F1 |
| ------- | --------- | ----------- |
| v0.0.1  | CAS/ESSAI | 79%         |
| v0.0.2  | CAS/ESSAI | 71%         |
| v0.0.2  | NegParHyp | 88%         |

The table shows that we overfit on the NegParHyp corpus.

## Authors and citation

The `negation` pipeline was developed by the Data Science team at EDS.

## References

```{eval-rst}
.. footbibliography::
```
