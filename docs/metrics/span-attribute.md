# Span Attribute Classification Metrics {: #edsnlp.metrics.span_attribute.SpanAttributeMetric }

Several NLP tasks consist in classifying existing spans of text into multiple classes,
such as the detection of negation, hypothesis or span linking. We provide a metric
to evaluate the performance of such tasks.

Let's look at an example. We'll use the following two documents: a reference
document (ref) and a document with predicted entities (pred).

+-------------------------------------------------------------------+-------------------------------------------------------------------+
| pred                                                              | ref                                                               |
+===================================================================+===================================================================+
| Le patient n'est pas *fièvreux*{.chip data-chip="SYMP neg=true"}, | Le patient n'est pas *fièvreux*{.chip data-chip="SYMP neg=true"}, |
| son père a *du diabète*{.chip data-chip="DIS carrier=PATIENT"}.   | son père a *du diabète*{.chip data-chip="DIS carrier=FATHER"}.    |
| Pas d'évolution du                                                | Pas d'évolution du                                                |
| *cancer*{.chip data-chip="DIS neg=true carrier=PATIENT"}.         | *cancer*{.chip data-chip="DIS carrier=PATIENT"}.                  |
+-------------------------------------------------------------------+-------------------------------------------------------------------+

We can quickly create matching documents in EDS-NLP using the following code snippet:

```python
from edsnlp.data.converters import MarkupToDocConverter

conv = MarkupToDocConverter(preset="md", span_setter="entities")
# Create a document with predicted attributes and a reference document
pred = conv(
    "Le patient n'est pas [fièvreux](SYMP neg=true), "
    "son père a [du diabète](DIS neg=false carrier=PATIENT). "
    "Pas d'évolution du [cancer](DIS neg=true carrier=PATIENT)."
)
ref = conv(
    "Le patient n'est pas [fièvreux](SYMP neg=true), "
    "son père a [du diabète](DIS neg=false carrier=FATHER). "
    "Pas d'évolution du [cancer](DIS neg=false carrier=PATIENT)."
)
```

::: edsnlp.metrics.span_attribute.SpanAttributeMetric
    options:
        heading_level: 2
        show_bases: false
        show_source: false
        only_class_level: true
