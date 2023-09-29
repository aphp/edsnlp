from collections import defaultdict
from typing import Any, Iterable

from spacy.training import Example

from edsnlp import registry
from edsnlp.scorers import make_examples
from edsnlp.utils.bindings import BINDING_GETTERS, Qualifiers, QualifiersArg
from edsnlp.utils.span_getters import SpanGetterArg, get_spans


def span_classification_scorer(
    examples: Iterable[Example],
    span_getter: SpanGetterArg,
    qualifiers: Qualifiers,
    include_falsy: bool = False,
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
        Whether to count predicted or gold occurences of falsy values when computing
        the metrics. If `False`, only the non-falsy values will be counted and matched
        together.

    Returns
    -------
    Dict[str, float]
    """
    labels = defaultdict(lambda: ([], []))
    labels[None] = ([], [])
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
                    labels[None][0].append((eg_idx, span_idx, qualifier, value))
                    key_str = f"{qualifier}" if value is True else f"{value}"
                    labels[key_str][0].append((eg_idx, span_idx, value))

        doc_spans = get_spans(eg.reference, span_getter)
        for span_idx, span in enumerate(doc_spans):
            total_gold_count += 1
            for qualifier, span_filter in qualifiers.items():
                if not (span_filter is True or span.label_ in span_filter):
                    continue
                value = BINDING_GETTERS[qualifier](span)
                if value or include_falsy:
                    labels[None][1].append((eg_idx, span_idx, qualifier, value))
                    key_str = f"{qualifier}" if value is True else f"{value}"
                    labels[key_str][1].append((eg_idx, span_idx, value))

    if total_pred_count != total_gold_count:
        raise ValueError(
            f"Number of predicted and gold spans differ: {total_pred_count} != "
            f"{total_gold_count}. Make sure that you are running your span "
            "qualifier pipe on the gold annotations, and not spans predicted by "
            "another NER pipe in your model."
        )

    def prf(pred, gold):
        tp = len(set(pred) & set(gold))
        np = len(pred)
        ng = len(gold)
        return {
            "f": 2 * tp / max(1, np + ng),
            "p": 1 if tp == np else (tp / np),
            "r": 1 if tp == ng else (tp / ng),
            "support": len(gold),
        }

    results = {name: prf(pred, gold) for name, (pred, gold) in labels.items()}
    micro_results = results.pop(None)
    return {
        "qual_p": micro_results["p"],
        "qual_r": micro_results["r"],
        "qual_f": micro_results["f"],
        "support": len(labels[None][1]),
        "qual_per_type": results,
    }


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
