from typing import Any, Callable, Dict, Iterable, List

from confit import Config
from spacy.tokens import Doc, Span
from spacy.training import Example

from edsnlp import registry
from edsnlp.core import PipelineProtocol
from edsnlp.core.component import BatchInput, TorchComponent

from ..embeddings.typing import WordEmbeddingBatchOutput
from .ner import CRFMode, TrainableNER, make_span_getter, nested_ner_exact_scorer

ner_default_config = """
[root]
mode = "joint"

[root.span_getter]
@misc = "span_getter"

[root.scorer]
@scorers = "eds.ner_exact_scorer"
"""

NER_DEFAULTS = Config.from_str(ner_default_config)["root"]


@registry.scorers.register("eds.ner_exact_scorer")
def create_ner_exact_scorer():
    return nested_ner_exact_scorer


@registry.factory.register(
    "eds.ner",
    default_config=NER_DEFAULTS,
    requires=["doc.ents", "doc.spans"],
    assigns=["doc.ents", "doc.spans"],
    default_score_weights={
        "ents_f": 1.0,
        "ents_p": 0.0,
        "ents_r": 0.0,
    },
)
def create_component(
    nlp: PipelineProtocol,
    name: str,
    embedding: TorchComponent[WordEmbeddingBatchOutput, BatchInput],
    labels: List[str] = [],
    span_getter: Callable[[Doc], Iterable[Span]] = make_span_getter(),
    mode: CRFMode = CRFMode.joint,
    scorer: Callable[[Iterable[Example]], Dict[str, Any]] = create_ner_exact_scorer(),
):
    """
    Initialize a general named entity recognizer (with or without nested or
    overlapping entities).

    Parameters
    ----------
    nlp: PipelineProtocol
        The current nlp object
    name: str
        Name of the component
    embedding: TorchComponent[WordEmbeddingBatchOutput, BatchInput]
        The word embedding component
    labels: List[str]
        The labels to predict. The labels can also be inferred from the data
        during `nlp.post_init(...)`
    span_getter: Callable[[Doc], Iterable[Span]]
        Method to call to get the gold spans from a document, for scoring or training
    mode: CRFMode
        The CRF mode to use: independent, joint or marginal
    scorer: Optional[Callable[[Iterable[Example]], Dict[str, Any]]]
        Method to call to score predictions
    """
    return TrainableNER(
        nlp=nlp,
        name=name,
        embedding=embedding,
        labels=labels,
        span_getter=span_getter,
        mode=mode,
        scorer=scorer,
    )
