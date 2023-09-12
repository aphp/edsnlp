# Named Entity Recognition Components

We provide several Named Entity Recognition (NER) components.
Named Entity Recognition is the task of identifying short relevant spans of text, named entities, and classifying them into pre-defined categories.
In the case of clinical documents, these entities can be scores, disorders, behaviors, codes, dates, measurements, etc.

## Span setters: where are stored extracted entities ? {: #edsnlp.pipelines.base.SpanSetterArg }

A component assigns entities to a document by adding them to the `doc.ents` or `doc.spans[group]` attributes. `doc.ents` only supports non overlapping
entities, therefore, if two entities overlap, the longest one will be kept. `doc.spans[group]` on the other hand, can contain overlapping entities.
To control where entities are added, you can use the `span_setter` argument in any of these component.

::: edsnlp.pipelines.base.SpanSetterArg
    options:
        heading_level: 2
        show_bases: false
        show_source: false
        only_class_level: true

## Available components

<!-- --8<-- [start:components] -->

| Component                                                                                 | Description                           |
|-------------------------------------------------------------------------------------------|---------------------------------------|
| [`eds.covid`](/pipelines/ner/covid)                                                       | A COVID mentions detector             |
| [`eds.charlson`](/pipelines/ner/scores/charlson)                                          | A Charlson score extractor            |
| [`eds.sofa`](/pipelines/ner/scores/sofa)                                                  | A SOFA score extractor                |
| [`eds.elston_ellis`](/pipelines/ner/scores/elston-ellis)                                  | An Elston & Ellis code extractor      |
| [`eds.emergency_priority`](/pipelines/ner/scores/emergency-priority)                      | A priority score extractor            |
| [`eds.emergency_ccmu`](/pipelines/ner/scores/emergency-ccmu)                              | A CCMU score extractor                |
| [`eds.emergency_gemsa`](/pipelines/ner/scores/emergency-gemsa)                            | A GEMSA score extractor               |
| [`eds.tnm`](/pipelines/ner/tnm)                                                           | A TNM score extractor                 |
| [`eds.adicap`](/pipelines/ner/adicap)                                                     | A ADICAP codes extractor              |
| [`eds.drugs`](/pipelines/ner/drugs)                                                       | A drug mentions extractor             |
| [`eds.cim10`](/pipelines/ner/cim10)                                                       | A CIM10 terminology matcher           |
| [`eds.umls`](/pipelines/ner/umls)                                                         | An UMLS terminology matcher           |
| [`eds.ckd`](/pipelines/ner/disorders/ckd)                                                 | CKD extractor                         |
| [`eds.copd`](/pipelines/ner/disorders/copd)                                               | COPD extractor                        |
| [`eds.cerebrovascular_accident`](/pipelines/ner/disorders/cerebrovascular-accident)       | Cerebrovascular accident extractor    |
| [`eds.congestive_heart_failure`](/pipelines/ner/disorders/congestive-heart-failure)       | Congestive heart failure extractor    |
| [`eds.connective_tissue_disease`](/pipelines/ner/disorders/connective-tissue-disease)     | Connective tissue disease extractor   |
| [`eds.dementia`](/pipelines/ner/disorders/dementia)                                       | Dementia extractor                    |
| [`eds.diabetes`](/pipelines/ner/disorders/diabetes)                                       | Diabetes extractor                    |
| [`eds.hemiplegia`](/pipelines/ner/disorders/hemiplegia)                                   | Hemiplegia extractor                  |
| [`eds.leukemia`](/pipelines/ner/disorders/leukemia)                                       | Leukemia extractor                    |
| [`eds.liver_disease`](/pipelines/ner/disorders/liver-disease)                             | Liver disease extractor               |
| [`eds.lymphoma`](/pipelines/ner/disorders/lymphoma)                                       | Lymphoma extractor                    |
| [`eds.myocardial_infarction`](/pipelines/ner/disorders/myocardial-infarction)             | Myocardial infarction extractor       |
| [`eds.peptic_ulcer_disease`](/pipelines/ner/disorders/peptic-ulcer-disease)               | Peptic ulcer disease extractor        |
| [`eds.peripheral_vascular_disease`](/pipelines/ner/disorders/peripheral-vascular-disease) | Peripheral vascular disease extractor |
| [`eds.solid_tumor`](/pipelines/ner/disorders/solid-tumor)                                 | Solid tumor extractor                 |
| [`eds.alcohol`](/pipelines/ner/behaviors/alcohol)                                         | Alcohol consumption extractor         |
| [`eds.tobacco`](/pipelines/ner/behaviors/tobacco)                                         | Tobacco consumption extractor         |

<!-- --8<-- [end:components] -->
