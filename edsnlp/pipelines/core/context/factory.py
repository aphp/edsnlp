from typing import List

from edsnlp.core import PipelineProtocol, registry
from edsnlp.pipelines.core.context import ContextAdder

DEFAULT_CONFIG = dict(
    context=["note_id"],
)


@registry.factory.register("eds.context")
def create_component(
    nlp: PipelineProtocol,
    name: str,
    *,
    context: List[str] = ["note_id"],
):

    return ContextAdder(
        nlp,
        context=context,
    )
