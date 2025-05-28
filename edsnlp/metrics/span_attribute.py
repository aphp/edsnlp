"""
Metrics for Span Attribute Classification

# Span Attribute Classification Metrics {: #edsnlp.metrics.span_attribute.SpanAttributeMetric }

Several NLP tasks consist in classifying existing spans of text into multiple classes,
such as the detection of negation, hypothesis or span linking.

We provide a metric to evaluate the performance of such tasks,

Let's look at an example:

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
"""  # noqa: E501

import warnings
from collections import defaultdict
from typing import Any, Dict, Optional

from edsnlp import registry
from edsnlp.metrics import Examples, average_precision, make_examples, prf
from edsnlp.utils.bindings import BINDING_GETTERS, Attributes, AttributesArg
from edsnlp.utils.span_getters import SpanGetterArg, get_spans


def span_attribute_metric(
    examples: Examples,
    span_getter: SpanGetterArg,
    attributes: Attributes = None,
    include_falsy: bool = False,
    default_values: Dict = {},
    micro_key: str = "micro",
    filter_expr: Optional[str] = None,
    **kwargs: Any,
):
    if "qualifiers" in kwargs:
        warnings.warn(
            "The `qualifiers` argument of span_attribute_metric() is "
            "deprecated. Use `attributes` instead.",
            DeprecationWarning,
        )
        assert attributes is None
        attributes = kwargs.pop("qualifiers")
    if attributes is None:
        raise TypeError(
            "span_attribute_metric() missing 1 required argument: 'attributes'"
        )
    if kwargs:
        raise TypeError(
            f"span_attribute_metric() got unexpected keyword arguments: "
            f"{', '.join(kwargs.keys())}"
        )
    examples = make_examples(examples)
    if filter_expr is not None:
        filter_fn = eval(f"lambda doc: {filter_expr}")
        examples = [eg for eg in examples if filter_fn(eg.reference)]
    labels = defaultdict(lambda: (set(), set(), dict()))
    labels["micro"] = (set(), set(), dict())
    total_pred_count = 0
    total_gold_count = 0

    if not include_falsy:
        default_values_ = defaultdict(lambda: False)
        default_values_.update(default_values)
        default_values = default_values_
        del default_values_
    for eg_idx, eg in enumerate(examples):
        doc_spans = get_spans(eg.predicted, span_getter)
        for span in doc_spans:
            total_pred_count += 1
            beg, end = span.start, span.end
            for attr, span_filter in attributes.items():
                if not (span_filter is True or span.label_ in span_filter):
                    continue
                getter_key = attr if attr.startswith("_.") else f"_.{attr}"
                value = BINDING_GETTERS[getter_key](span)
                top_val, top_p = max(
                    getattr(span._, "prob", {}).get(attr, {}).items(),
                    key=lambda x: x[1],
                    default=(value, 1.0),
                )
                if (top_val or include_falsy) and default_values[attr] != top_val:
                    labels[attr][2][(eg_idx, beg, end, attr, top_val)] = top_p
                    labels[micro_key][2][(eg_idx, beg, end, attr, top_val)] = top_p
                if (value or include_falsy) and default_values[attr] != value:
                    labels[micro_key][0].add((eg_idx, beg, end, attr, value))
                    labels[attr][0].add((eg_idx, beg, end, attr, value))

        doc_spans = get_spans(eg.reference, span_getter)
        for span in doc_spans:
            total_gold_count += 1
            beg, end = span.start, span.end
            for attr, span_filter in attributes.items():
                if not (span_filter is True or span.label_ in span_filter):
                    continue
                getter_key = attr if attr.startswith("_.") else f"_.{attr}"
                value = BINDING_GETTERS[getter_key](span)
                if (value or include_falsy) and default_values[attr] != value:
                    labels[micro_key][1].add((eg_idx, beg, end, attr, value))
                    labels[attr][1].add((eg_idx, beg, end, attr, value))

    if total_pred_count != total_gold_count:
        raise ValueError(
            f"Number of predicted and gold spans differ: {total_pred_count} != "
            f"{total_gold_count}. Make sure that you are running your span "
            "attribute classification pipe on the gold annotations, and not spans "
            "predicted by another NER pipe in your model."
        )

    for name, (pred, gold, pred_with_prob) in labels.items():
        print("-", name, "pred/gold", pred, gold, "=>", prf(pred, gold))
    return {
        name: {
            **prf(pred, gold),
            "ap": average_precision(pred_with_prob, gold),
        }
        for name, (pred, gold, pred_with_prob) in labels.items()
    }


@registry.metrics.register(
    "eds.span_attribute",
    deprecated=["eds.span_classification_scorer", "eds.span_attribute_scorer"],
)
class SpanAttributeMetric:
    """
    The `eds.span_attribute` metric
    evaluates span‐level attribute classification by comparing predicted and gold
    attribute values on the same set of spans. For each attribute you specify, it
    computes Precision, Recall, F1, number of true positives (tp), number of
    gold instances (support), number of predicted instances (positives), and
    the Average Precision (ap). A micro‐average over all attributes is also
    provided under `micro_key`.

    ```python
    from edsnlp.metrics.span_attribute import SpanAttributeMetric

    metric = SpanAttributeMetric(
        span_getter=conv.span_setter,
        # Evaluated attributes
        attributes={
            "neg": True,  # 'neg' on every entity
            "carrier": ["DIS"],  # 'carrier' only on 'DIS' entities
        },
        # Ignore these default values when counting matches
        default_values={
            "neg": False,
        },
        micro_key="micro",
    )
    ```

    Let's enumerate (span -> attr = value) items in our documents. Only the items with
    matching span boundaries, attribute name, and value are counted as a true positives.
    For instance, with the predicted and reference spans of the example above:

    +--------------------------------------------------+-------------------------------------------------+
    | pred                                             | ref                                             |
    +==================================================+=================================================+
    | *fièvreux → neg = True*{.chip .tp}<br/>          | *fièvreux → neg = True*{.chip .tp}<br/>         |
    | *du diabète → neg = False*{.chip .na}<br/>       | *du diabète → neg = False*{.chip .na}<br/>      |
    | *du diabète → carrier = PATIENT*{.chip .fp}<br/> | *du diabète → carrier = FATHER*{.chip .fn}<br/> |
    | *cancer → neg = True*{.chip .fp}<br/>            | *cancer → neg = False*{.chip .na}<br/>          |
    | *cancer → carrier = PATIENT*{.chip .tp}          | *cancer → carrier = PATIENT*{.chip .tp}         |
    +--------------------------------------------------+-------------------------------------------------+

    !!! note "Default values"

        Note that there we don't count "neg=False" items, shown in grey in the table. In EDS-NLP,
        this is done by setting `defaults_values={"neg": False}` when creating the metric. This
        is quite common in classification tasks, where one of the values is both the most common
        and the "default" (hence the name of the parameter). Counting these values would likely
        skew the micro-average metrics towards the default value.

    Precision, Recall and F1 (micro-average and per‐label) are computed as follows:

    - Precision: `p = |matched items of pred| / |pred|`
    - Recall: `r = |matched items of ref| / |ref|`
    - F1: `f = 2 / (1/p + 1/f)`

    This yields the following metrics:

    ```python
    metric([ref], [pred])
    # Out: {
    #   'micro': {'f': 0.57, 'p': 0.5, 'r': 0.67, 'tp': 2, 'support': 3, 'positives': 4, 'ap': 0.17},
    #   'neg': {'f': 0.67, 'p': 0.5, 'r': 1, 'tp': 1, 'support': 1, 'positives': 2, 'ap': 0.0},
    #   'carrier': {'f': 0.5, 'p': 0.5, 'r': 0.5, 'tp': 1, 'support': 2, 'positives': 2, 'ap': 0.25},
    # }
    ```

    Parameters
    ----------
    span_getter : SpanGetterArg
        The span getter to extract spans from each `Doc`.
    attributes : Mapping[str, Union[bool, Sequence[str]]]
        Map each attribute name to `True` (evaluate on all spans) or a sequence of
        labels restricting which spans to test.
    default_values : Dict[str, Any]
        Attribute values to omit from micro‐average counts (e.g., common negative or
        default labels).
    include_falsy : bool
        If `False`, ignore falsy values (e.g., `False`, `None`, `''`) in predictions
        or gold when computing metrics; if `True`, count them.
    micro_key : str
        Key under which to store the micro‐averaged results across all attributes.
    filter_expr : Optional[str]
        A Python expression (using `doc`) to filter which examples are scored.

    Returns
    -------
    Dict[str, Dict[str, float]]
        A dictionary mapping each attribute name (and the `micro_key`) to its metrics:

        - `label` or micro_key :

            - `p` : precision
            - `r` : recall
            - `f` : F1 score
            - `tp` : true positive count
            - `support` : number of gold instances
            - `positives` : number of predicted instances
            - `ap` : [average precision](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)
    """  # noqa: E501

    attributes: Attributes

    def __init__(
        self,
        span_getter: SpanGetterArg,
        attributes: AttributesArg = None,
        qualifiers: AttributesArg = None,
        default_values: Dict = {},
        include_falsy: bool = False,
        micro_key: str = "micro",
        filter_expr: Optional[str] = None,
    ):
        if qualifiers is not None:
            warnings.warn(
                "The `qualifiers` argument is deprecated. Use `attributes` instead.",
                DeprecationWarning,
            )
        self.span_getter = span_getter
        self.attributes = attributes or qualifiers
        self.default_values = default_values
        self.include_falsy = include_falsy
        self.micro_key = micro_key
        self.filter_expr = filter_expr

    __init__.__doc__ = span_attribute_metric.__doc__

    def __call__(self, *examples: Any):
        """
        Compute the span attribute metrics for the given examples.

        Parameters
        ----------
        examples : Examples
            The examples to score, either a tuple of (golds, preds) or a list of
            spacy.training.Example objects

        Returns
        -------
        Dict[str, Dict[str, float]]
            The scores for the attributes
        """
        return span_attribute_metric(
            examples,
            span_getter=self.span_getter,
            attributes=self.attributes,
            default_values=self.default_values,
            include_falsy=self.include_falsy,
            micro_key=self.micro_key,
            filter_expr=self.filter_expr,
        )


# For backward compatibility
span_classification_scorer = span_attribute_scorer = span_attribute_metric
create_span_attributes_scorer = SpanAttributeScorer = SpanAttributeMetric

__all__ = [
    "span_attribute_metric",
    "SpanAttributeMetric",
]
