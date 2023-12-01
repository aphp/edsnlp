# Aggregating results

## Rationale

In some cases, you are not interested in individual extractions, but rather in document-level aggregated variables. For instance, you may be interested to know if a patient is diabetic without caring abou the actual mentions of diabetes. Here, we propose a simple and generic rule which work by:

- Extracting entities via methods of your choice
- Qualifiy those entities and discard appropriate entities
- Set a threshold on the minimal number of entities that should be present in the document to aggregate them.


## An example for the [disorders][disorders] pipes

Below is a simple implementation of this aggregation rule (this can be adapted for other comorbidity components and other qualification methods):

```{ .python .no-check }
MIN_NUMBER_ENTITIES = 2  # (1)!

if not Doc.has_extension("aggregated"):
    Doc.set_extension("aggregated", default={})  # (2)!

spans = doc.spans["diabetes"]  # (3)!
kept_spans = [
    (span, span._.status, span._.detailed_status)
    for span in spans
    if not any([span._.negation, span._.hypothesis, span._.family])
]  # (4)!

if len(kept_spans) < MIN_NUMBER_ENTITIES:  # (5)!
    status = "ABSENT"

else:
    status = max(kept_spans, key=itemgetter(1))[2]  # (6)!

doc._.aggregated["diabetes"] = status
```

1. We want at least 2 correct entities
2. Storing the status in the `doc._.aggregated` dictionary
3. Getting status for the `diabetes` component
4. Disregarding entities which are either negated, hypothetical, or not
about the patient himself
5. Setting the status to 0 if less than 2 relevant entities are left:
6. Getting the maximum severity status
