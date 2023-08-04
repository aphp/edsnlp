# Pipelines overview

EDS-NLP provides easy-to-use spaCy components.

=== "Core"

    | Pipeline                 | Description                                     |
    | ------------------------ | ----------------------------------------------- |
    | `eds.normalizer`         | Non-destructive input text normalisation        |
    | `eds.sentences`          | Better sentence boundary detection              |
    | `eds.matcher`            | A simple yet powerful entity extractor          |
    | `eds.terminology`        | A simple yet powerful terminology matcher       |
    | `eds.contextual-matcher` | A conditional entity extractor                  |
    | `eds.endlines`           | An unsupervised model to classify each end line |

=== "Qualifiers"

    | Pipeline              | Description                          |
    | --------------------- | ------------------------------------ |
    | `eds.negation`        | Rule-based negation detection        |
    | `eds.family`          | Rule-based family context detection  |
    | `eds.hypothesis`      | Rule-based speculation detection     |
    | `eds.reported_speech` | Rule-based reported speech detection |
    | `eds.history`         | Rule-based medical history detection |

=== "Miscellaneous"

    | Pipeline                 | Description                                 |
    | ------------------------ | ------------------------------------------- |
    | `eds.dates`              | Date extraction and normalisation           |
    | `eds.consultation_dates` | Identify consultation dates                 |
    | `eds.measurements`       | Measure extraction and normalisation        |
    | `eds.sections`           | Section detection                           |
    | `eds.reason`             | Rule-based hospitalisation reason detection |
    | `eds.tables`             | Tables detection                            |

=== "NER"

    | Pipeline                          | Description                           |
    | --------------------------------- | ------------------------------------- |
    | `eds.covid`                       | A COVID mentions detector             |
    | `eds.charlson`                    | A Charlson score extractor            |
    | `eds.elstonellis`                 | An Elston & Ellis code extractor      |
    | `eds.emergency.priority`          | A priority score extractor            |
    | `eds.emergency.ccmu`              | A CCMU score extractor                |
    | `eds.emergency.gemsa`             | A GEMSA score extractor               |
    | `eds.sofa`                        | A SOFA score extractor                |
    | `eds.tnm`                         | A TNM score extractor                 |
    | `eds.adicap`                      | A ADICAP codes extractor              |
    | `eds.drugs`                       | A drug mentions extractor             |
    | `eds.cim10`                       | A CIM10 terminology matcher           |
    | `eds.umls`                        | An UMLS terminology matcher           |
    | `eds.ckd`                         | CKD extractor                         |
    | `eds.copd`                        | COPD extractor                        |
    | `eds.cerebrovascular_accident`    | Cerebrovascular accident extractor    |
    | `eds.congestive_heart_failure`    | Congestive heart failure extractor    |
    | `eds.connective_tissue_disease`   | Connective tissue disease extractor   |
    | `eds.dementia`                    | Dementia extractor                    |
    | `eds.diabetes`                    | Diabetes extractor                    |
    | `eds.hemiplegia`                  | Hemiplegia extractor                  |
    | `eds.leukemia`                    | Leukemia extractor                    |
    | `eds.liver_disease`               | Liver disease extractor               |
    | `eds.lymphoma`                    | Lymphoma extractor                    |
    | `eds.myocardial_infarction`       | Myocardial infarction extractor       |
    | `eds.peptic_ulcer_disease`        | Peptic ulcer disease extractor        |
    | `eds.peripheral_vascular_disease` | Peripheral vascular disease extractor |
    | `eds.solid_tumor`                 | Solid tumor extractor                 |
    | `eds.alcohol`                     | Alcohol consumption extractor         |
    | `eds.tobacco`                     | Tobacco consumption extractor         |

=== "Trainable"

    | Pipeline             | Description                                                          |
    | -------------------- | -------------------------------------------------------------------- |
    | `eds.nested-ner`     | A trainable component for nested (and classic) NER                   |
    | `eds.span-qualifier` | A trainable component for multi-class multi-label span qualification |

You can add them to your spaCy pipeline by simply calling `add_pipe`, for instance:

<!-- no-check -->

```python
# ↑ Omitted code that defines the nlp object ↑
nlp.add_pipe("eds.normalizer")
```
