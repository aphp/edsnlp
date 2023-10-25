from collections import defaultdict
from typing import Any, Iterable

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
    for eg_idx, eg in enumerate(examples):
        doc_spans = get_spans(eg.predicted, span_getter)
        for span_idx, span in enumerate(doc_spans):
            total_pred_count += 1
            for qualifier, span_filter in qualifiers.items():
                if not (span_filter is True or span.label_ in span_filter):
                    continue
                value = BINDING_GETTERS[qualifier](span)
                if value or include_falsy:
                    labels[micro_key][0].add((eg_idx, span_idx, qualifier, value))
                    labels[qualifier][0].add((eg_idx, span_idx, qualifier, value))

        doc_spans = get_spans(eg.reference, span_getter)
        for span_idx, span in enumerate(doc_spans):
            total_gold_count += 1
            for qualifier, span_filter in qualifiers.items():
                if not (span_filter is True or span.label_ in span_filter):
                    continue
                value = BINDING_GETTERS[qualifier](span)
                if value or include_falsy:
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
    ):
        self.span_getter = span_getter
        self.qualifiers = qualifiers  # type: ignore

    def __call__(self, *examples: Any):
        return span_classification_scorer(
            make_examples(*examples),
            self.span_getter,
            self.qualifiers,
        )
