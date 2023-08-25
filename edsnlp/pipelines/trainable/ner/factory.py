from typing import Dict, Optional, Union

from pydantic.types import StrictStr
from typing_extensions import Literal

from edsnlp import registry
from edsnlp.core import PipelineProtocol
from edsnlp.core.component import BatchInput, TorchComponent
from edsnlp.utils.span_getters import ListStr, SpanGetter

from ..embeddings.typing import WordEmbeddingBatchOutput
from .ner import INFER, TrainableNER


@registry.factory.register(
    "eds.ner",
    requires=["doc.ents", "doc.spans"],
    assigns=["doc.ents", "doc.spans"],
    default_score_weights={
        "ents_f": 1.0,
        "ents_p": 0.0,
        "ents_r": 0.0,
    },
)
def create_component(
    nlp: Optional[PipelineProtocol] = None,
    name: Optional[str] = None,
    *,
    embedding: TorchComponent[WordEmbeddingBatchOutput, BatchInput],
    to_ents: Union[bool, ListStr] = INFER,
    to_span_groups: Union[StrictStr, Dict[str, Union[bool, ListStr]]] = INFER,
    labels: Optional[ListStr] = INFER,
    target_span_getter: SpanGetter = {"ents": True},
    mode: Literal["independent", "joint", "marginal"] = "joint",
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
    target_span_getter: Callable[[Doc], Iterable[Span]]
        Method to call to get the gold spans from a document, for scoring or training.
        By default, takes all entities in `doc.ents`, but we recommend you specify
        a given span group name instead.
    labels: List[str]
        The labels to predict. The labels can also be inferred from the data
        during `nlp.post_init(...)`
    to_ents: ListStrOrBool
        Whether to put predictions in `doc.ents`. `to_ents` can be:
            - a boolean to put all or no predictions in `doc.ents`
            - a list of str to filter predictions by label
    to_span_groups: Union[str, Dict[str, ListStrOrBool]]
        If and how to put predictions in `doc.spans`. `to_span_groups` can be:
            - a string to put all predictions to a given span group (e.g. "ner-preds")
            - a dict mapping group names to a list of str to filter predictions by label
    mode: Literal["independent", "joint", "marginal"]
        The CRF mode to use: independent, joint or marginal
    """
    return TrainableNER(
        nlp=nlp,
        name=name,
        embedding=embedding,
        to_ents=to_ents,
        to_span_groups=to_span_groups,
        labels=labels,
        target_span_getter=target_span_getter,
        mode=mode,
    )
