from typing import List

from spacy.language import Language

from edsnlp.pipelines.core.context import ContextAdder

DEFAULT_CONFIG = dict(
    context=["note_id"],
)


@Language.factory(
    "eds.context",
    default_config=DEFAULT_CONFIG,
)
def create_component(
    nlp: Language,
    name: str,
    context: List[str],
):

    return ContextAdder(
        nlp,
        context=context,
    )
