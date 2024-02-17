from collections import defaultdict
from typing import Any, Dict, Iterable

from spacy.training import Example

from edsnlp import registry
from edsnlp.scorers import make_examples, prf
from edsnlp.utils.bindings import BINDING_GETTERS, Qualifiers, QualifiersArg
from edsnlp.utils.span_getters import SpanGetterArg, get_spans


def span_classification_scorer(
    examples: Iterable[Example],
    span_getter: SpanGetterArg,
    qualifiers: Qualifiers,
    include_falsy: bool = False,
    default_values: Dict = {},
    micro_key: str = "micro",
):
    """
    Scores the extracted entities that may be overlapping or nested
    by looking in `doc.ents`, and `doc.spans`.

    Parameters
    ----------
    examples : Iterable[Example]
        The examples to score
    span_getter : SpanGetterArg
        The span getter to use to extract the spans from the document
    qualifiers : Sequence[str]
        The qualifiers to use to score the spans
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

    Returns
    -------
    Dict[str, float]
    """
    labels = defaultdict(lambda: (set(), set()))
    labels["micro"] = (set(), set())
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
            for qualifier, span_filter in qualifiers.items():
                if not (span_filter is True or span.label_ in span_filter):
                    continue
                getter_key = (
                    qualifier if qualifier.startswith("_.") else f"_.{qualifier}"
                )
                value = BINDING_GETTERS[getter_key](span)
                if (value or include_falsy) and default_values[qualifier] != value:
                    labels[micro_key][0].add((eg_idx, span_idx, qualifier, value))
                    labels[qualifier][0].add((eg_idx, span_idx, qualifier, value))

        doc_spans = get_spans(eg.reference, span_getter)
        for span_idx, span in enumerate(doc_spans):
            total_gold_count += 1
            for qualifier, span_filter in qualifiers.items():
                if not (span_filter is True or span.label_ in span_filter):
                    continue
                getter_key = (
                    qualifier if qualifier.startswith("_.") else f"_.{qualifier}"
                )
                value = BINDING_GETTERS[getter_key](span)
                if (value or include_falsy) and default_values[qualifier] != value:
                    labels[micro_key][1].add((eg_idx, span_idx, qualifier, value))
                    labels[qualifier][1].add((eg_idx, span_idx, qualifier, value))

    if total_pred_count != total_gold_count:
        raise ValueError(
            f"Number of predicted and gold spans differ: {total_pred_count} != "
            f"{total_gold_count}. Make sure that you are running your span "
            "qualifier pipe on the gold annotations, and not spans predicted by "
            "another NER pipe in your model."
        )

    return {name: prf(pred, gold) for name, (pred, gold) in labels.items()}


@registry.scorers.register("eds.span_classification_scorer")
class create_span_classification_scorer:
    qualifiers: Qualifiers

    def __init__(
        self,
        span_getter: SpanGetterArg,
        qualifiers: QualifiersArg = None,
        default_values: Dict = {},
        include_falsy: bool = False,
        micro_key: str = "micro",
    ):
        self.span_getter = span_getter
        self.qualifiers = qualifiers
        self.default_values = default_values
        self.include_falsy = include_falsy
        self.micro_key = micro_key

    def __call__(self, *examples: Any):
        return span_classification_scorer(
            make_examples(*examples),
            span_getter=self.span_getter,
            qualifiers=self.qualifiers,
            default_values=self.default_values,
            include_falsy=self.include_falsy,
            micro_key=self.micro_key,
        )
