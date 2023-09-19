from collections import defaultdict
from typing import Iterable

from spacy.training import Example

from edsnlp import registry
from edsnlp.scorers import make_examples
from edsnlp.utils.bindings import BINDING_GETTERS, BindingCandidateGetterArg
from edsnlp.utils.span_getters import SpanGetterArg


def span_classification_scorer(
    examples: Iterable[Example],
    candidate_getter: BindingCandidateGetterArg,
):
    """
    Scores the extracted entities that may be overlapping or nested
    by looking in `doc.ents`, and `doc.spans`.

    Parameters
    ----------
    examples : Iterable[Example]
        The examples to score
    candidate_getter : BindingCandidateGetterArg
        The configuration dict of the component

    Returns
    -------
    Dict[str, float]
    """
    labels = defaultdict(lambda: ([], []))
    labels[None] = ([], [])
    for eg_idx, eg in enumerate(examples):
        doc_spans, *_, doc_qlf = candidate_getter(eg.predicted)
        for span_idx, (span, span_qualifiers) in enumerate(zip(doc_spans, doc_qlf)):
            for qualifier in span_qualifiers:
                value = BINDING_GETTERS[qualifier](span)
                if value:
                    labels[None][0].append((eg_idx, span_idx, qualifier, value))
                    key_str = f"{qualifier}" if value is True else f"{value}"
                    labels[key_str][0].append((eg_idx, span_idx, value))

        doc_spans, *_, doc_qlf = candidate_getter(eg.reference)
        for span_idx, (span, span_qualifiers) in enumerate(zip(doc_spans, doc_qlf)):
            for qualifier in span_qualifiers:
                value = BINDING_GETTERS[qualifier](span)
                if value:
                    labels[None][1].append((eg_idx, span_idx, qualifier, value))
                    key_str = f"{qualifier}" if value is True else f"{value}"
                    labels[key_str][1].append((eg_idx, span_idx, value))

    def prf(pred, gold):
        tp = len(set(pred) & set(gold))
        np = len(pred)
        ng = len(gold)
        return {
            "f": 2 * tp / max(1, np + ng),
            "p": 1 if tp == np else (tp / np),
            "r": 1 if tp == ng else (tp / ng),
        }

    results = {name: prf(pred, gold) for name, (pred, gold) in labels.items()}
    return {
        "qual_f": results[None]["f"],
        "qual_p": results[None]["p"],
        "qual_r": results[None]["r"],
        "qual_per_type": results,
    }


@registry.scorers.register("eds.span_classification_scorer")
def create_span_classification_scorer(
    span_getter: SpanGetterArg,
):
    return lambda *args: span_classification_scorer(make_examples(*args), span_getter)
