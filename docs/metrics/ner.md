# NER Metrics

We provide several metrics to evaluate the performance of Named Entity Recognition (NER) components.
Let's look at an example and see how they differ. We'll use the following two documents: a reference
document (ref) and a document with predicted entities (pred).

+-------------------------------------------------------------+------------------------------------------+
| pred                                                        | ref                                      |
+=============================================================+==========================================+
| *La*{.chip data-chip=PER} *patiente*{.chip data-chip=PER} a | La *patiente*{.chip data-chip=PER} a     |
| une *fièvre aigüe*{.chip data-chip=DIS}                     | *une fièvre*{.chip data-chip=DIS} aigüe. |
+-------------------------------------------------------------+------------------------------------------+

Let's create matching documents in EDS-NLP using the following code snippet:

```python
from edsnlp.data.converters import MarkupToDocConverter

conv = MarkupToDocConverter(preset="md", span_setter="entities")

pred = conv("[La](PER) [patiente](PER) a une [fièvre aiguë](DIS).")
ref = conv("La [patiente](PER) a [une fièvre](DIS) aiguë.")
```

## Span-level NER metric with exact match {: #edsnlp.metrics.ner.NerExactMetric }

::: edsnlp.metrics.ner.NerExactMetric
    options:
        heading_level: 2
        show_bases: false
        show_source: false
        only_class_level: true

## Span-level NER metric with approximate match {: #edsnlp.metrics.ner.NerOverlapMetric }

::: edsnlp.metrics.ner.NerOverlapMetric
    options:
        heading_level: 2
        show_bases: false
        show_source: false
        only_class_level: true


## Token-level NER metric {: #edsnlp.metrics.ner.NerTokenMetric }

::: edsnlp.metrics.ner.NerTokenMetric
    options:
        heading_level: 2
        show_bases: false
        show_source: false
        only_class_level: true
