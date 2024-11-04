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
    """
    Scores the attributes predictions between a list of gold and predicted spans.

    Parameters
    ----------
    examples : Examples
        The examples to score, either a tuple of (golds, preds) or a list of
        spacy.training.Example objects
    span_getter : SpanGetterArg
        The span getter to use to extract the spans from the document
    attributes : Sequence[str]
        The attributes to use to score the spans
    default_values: Dict
        Values to dismiss when computing the micro-average per label. This is
        useful to compute precision and recall for certain attributes that have
        imbalanced value repartitions, such as "negation", "family related"
        or "certainty" attributes.
    include_falsy : bool
        Whether to count predicted or gold occurrences of falsy values when computing
        the metrics. If `False`, only the non-falsy values will be counted and matched
        together.
    micro_key : str
        The key to use to store the micro-averaged results for spans of all types
    filter_expr : Optional[str]
        The filter expression to use to filter the documents

    Returns
    -------
    Dict[str, float]
    """
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
        for span_idx, span in enumerate(doc_spans):
            total_pred_count += 1
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
                    labels[attr][2][(eg_idx, span_idx, attr, top_val)] = top_p
                    labels[micro_key][2][(eg_idx, span_idx, attr, top_val)] = top_p
                if (value or include_falsy) and default_values[attr] != value:
                    labels[micro_key][0].add((eg_idx, span_idx, attr, value))
                    labels[attr][0].add((eg_idx, span_idx, attr, value))

        doc_spans = get_spans(eg.reference, span_getter)
        for span_idx, span in enumerate(doc_spans):
            total_gold_count += 1
            for attr, span_filter in attributes.items():
                if not (span_filter is True or span.label_ in span_filter):
                    continue
                getter_key = attr if attr.startswith("_.") else f"_.{attr}"
                value = BINDING_GETTERS[getter_key](span)
                if (value or include_falsy) and default_values[attr] != value:
                    labels[micro_key][1].add((eg_idx, span_idx, attr, value))
                    labels[attr][1].add((eg_idx, span_idx, attr, value))

    if total_pred_count != total_gold_count:
        raise ValueError(
            f"Number of predicted and gold spans differ: {total_pred_count} != "
            f"{total_gold_count}. Make sure that you are running your span "
            "attribute classification pipe on the gold annotations, and not spans "
            "predicted by another NER pipe in your model."
        )

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
