# Named Entity Recognition Components

We provide several Named Entity Recognition (NER) components.
Named Entity Recognition is the task of identifying short relevant spans of text, named entities, and classifying them into pre-defined categories.
In the case of clinical documents, these entities can be scores, disorders, behaviors, codes, dates, quantities, etc.

## Span setters: where are stored extracted entities ? {: #edsnlp.pipes.base.SpanSetterArg }

A component assigns entities to a document by adding them to the `doc.ents` or `doc.spans[group]` attributes. `doc.ents` only supports non overlapping
entities, therefore, if two entities overlap, the longest one will be kept. `doc.spans[group]` on the other hand, can contain overlapping entities.
To control where entities are added, you can use the `span_setter` argument in any of these component.

::: edsnlp.pipes.base.SpanSetterArg
    options:
        heading_level: 2
        show_bases: false
        show_source: false
        only_class_level: true

## Available components

<!-- --8<-- [start:components] -->

| Component                              | Description                                      |
| -------------------------------------- | ------------------------------------------------ |
| `eds.covid`                            | A COVID mentions detector                        |
| `eds.charlson`                         | A Charlson score extractor                       |
| `eds.sofa`                             | A SOFA score extractor                           |
| `eds.elston_ellis`                     | An Elston & Ellis code extractor                 |
| `eds.emergency_priority`               | A priority score extractor                       |
| `eds.emergency_ccmu`                   | A CCMU score extractor                           |
| `eds.emergency_gemsa`                  | A GEMSA score extractor                          |
| `eds.tnm`                              | A TNM score extractor                            |
| `eds.adicap`                           | A ADICAP codes extractor                         |
| `eds.drugs`                            | A drug mentions extractor                        |
| `eds.cim10`                            | A CIM10 terminology matcher                      |
| `eds.umls`                             | An UMLS terminology matcher                      |
| `eds.ckd`                              | CKD extractor                                    |
| `eds.copd`                             | COPD extractor                                   |
| `eds.cerebrovascular_accident`         | Cerebrovascular accident extractor               |
| `eds.congestive_heart_failure`         | Congestive heart failure extractor               |
| `eds.connective_tissue_disease`        | Connective tissue disease extractor              |
| `eds.dementia`                         | Dementia extractor                               |
| `eds.diabetes`                         | Diabetes extractor                               |
| `eds.hemiplegia`                       | Hemiplegia extractor                             |
| `eds.leukemia`                         | Leukemia extractor                               |
| `eds.liver_disease`                    | Liver disease extractor                          |
| `eds.lymphoma`                         | Lymphoma extractor                               |
| `eds.myocardial_infarction`            | Myocardial infarction extractor                  |
| `eds.peptic_ulcer_disease`             | Peptic ulcer disease extractor                   |
| `eds.peripheral_vascular_disease`      | Peripheral vascular disease extractor            |
| `eds.solid_tumor`                      | Solid tumor extractor                            |
| `eds.alcohol`                          | Alcohol consumption extractor                    |
| `eds.tobacco`                          | Tobacco consumption extractor                    |
| `eds.functional_status`                | Functional status mentions extractor             |
| `eds.cognitive_status`                 | Cognitive status extractor                       |
| `eds.frailty_mentions`                 | Frailty mentions extractor                       |
| `eds.global_health_status`             | General status extractor                         |
| `eds.geriatric_assessment`             | Geriatric assessment mentions extractor          |
| `eds.incontinence_status`              | Incontinence status mentions extractor           |
| `eds.mobility_status`                  | Mobility mentions extractor                      |
| `eds.nutritional_status`               | Nutritional status extractor                     |
| `eds.pain_status`                      | Pain status mentions extractor                   |
| `eds.polypharmacy_status`              | Polypharmacy mentions extractor                  |
| `eds.sensory_status`                   | Sensory status extractor                         |
| `eds.social_status`                    | Social status extractor                          |
| `eds.psychological_status`             | Psychological status extractor                   |
| `eds.adl_score`                        | ADL score extractor                              |
| `eds.iadl_score`                       | IADL score extractor                             |
| `eds.bref_score`                       | BREF score extractor                             |
| `eds.chair_stand_score`                | Chair stand score extractor                      |
| `eds.pain_rating_score`                | EN / EVA pain rating scores extractor            |
| `eds.g8_score`                         | G8 score extractor                               |
| `eds.gait_speed_score`                 | Gait speed score extractor                       |
| `eds.geriatric_depression_scale_score` | Geriatric Depression Scale (GDS) score extractor |
| `eds.mini_gds_score`                   | Mini-GDS score extractor                         |
| `eds.mini_cog_score`                   | Mini-Cog score extractor                         |
| `eds.mini_mental_state_score`          | Mini mental state (MMS) score extractor          |
| `eds.moca_score`                       | MOCA score extractor                             |
| `eds.ecog_performance_status_score`    | Ecog-Performance status score extractor          |
| `eds.clinical_frailty_scale_score`     | Clinical frailty scale score extractor           |
| `eds.sppb_score`                       | SPPB score extractor                             |
| `eds.timed_up_and_go_score`            | Timed Up and Go score extractor                  |

<!-- --8<-- [end:components] -->
