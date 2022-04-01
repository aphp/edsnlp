# Reason

The `eds.reason` pipeline uses a rule-based algorithm to detect spans that relate to the reason of the hospitalisation. It was designed at AP-HP's EDS.

## Usage

The following snippet matches a simple terminology, and looks for spans of hospitalisation reasons. It is complete and can be run _as is_.

```python
import spacy

text = """COMPTE RENDU D'HOSPITALISATION du 11/07/2018 au 12/07/2018
MOTIF D'HOSPITALISATION
Monsieur Dupont Jean Michel, de sexe masculin, âgée de 39 ans, née le 23/11/1978, a été
hospitalisé du 11/08/2019 au 17/08/2019 pour attaque d'asthme.

ANTÉCÉDENTS
Antécédents médicaux :
Premier épisode d'asthme en mai 2018."""

nlp = spacy.blank("fr")

# Extraction of entities
nlp.add_pipe(
    "eds.matcher",
    config=dict(
        terms=dict(
            respiratoire=[
                "asthmatique",
                "asthme",
                "toux",
            ]
        )
    ),
)


nlp.add_pipe("eds.normalizer")
nlp.add_pipe("eds.reason", config=dict(use_sections=True))
doc = nlp(text)

reason = doc.spans["reasons"][0]
reason
# Out: hospitalisé du 11/08/2019 au 17/08/2019 pour attaque d'asthme.

reason._.is_reason
# Out: True

entities = reason._.ents_reason
entities
# Out: [asthme]

entities[0].label_
# Out: 'respiratoire'

ent = entities[0]
ent._.is_reason
# Out: True
```

## Configuration

The pipeline can be configured using the following parameters :

| Parameter         | Explanation                                      | Default                           |
| ----------------- | ------------------------------------------------ | --------------------------------- |
| `reasons`         | Reasons patterns                                 | `None` (use pre-defined patterns) |
| `attr`            | spaCy attribute to match on, eg `NORM` or `TEXT` | `"NORM"`                          |
| `use_sections`    | Whether to use sections                          | `False`                           |
| `ignore_excluded` | Whether to ignore excluded tokens                | `False`                           |

## Declared extensions

The `eds.reason` pipeline adds the key `reasons` to `doc.spans` and declares one [spaCy extension](https://spacy.io/usage/processing-pipelines#custom-components-attributes), on the `Span` objects called `ents_reason`.

The `ents_reason` extension is a list of named entities that overlap the `Span`, typically entities found in previous pipelines like `matcher`.

It also declares the boolean extension `is_reason`. This extension is set to True for the Reason Spans but also for the entities that overlap the reason span.

## Authors and citation

The `eds.reason` pipeline was developed by AP-HP's Data Science team.
