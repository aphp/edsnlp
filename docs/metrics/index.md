# Metrics

EDS-NLP provides several metrics to evaluate the performance of its components. These metrics can be used to assess the quality of entity recognition, negation detection, and other tasks.

At the moment, we support the following metrics:

| Metric               | Description                                        |
|:---------------------|:---------------------------------------------------|
| `eds.ner_exact`      | NER metric with exact match at the span level      |
| `eds.ner_token`      | NER metric with token-level match                  |
| `eds.ner_overlap`    | NER metric with overlap match at the span level    |
| `eds.span_attribute` | Span multi-label multi-class classification metric |
