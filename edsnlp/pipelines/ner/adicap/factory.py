from typing import List, Optional, Union

from spacy.language import Language

from .adicap import Adicap

DEFAULT_CONFIG = dict(
    pattern=None,
    attr="TEXT",
)


@Language.factory(
    "eds.adicap",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    pattern: Optional[Union[List[str], str]],
    attr: str,
):
    return Adicap(
        nlp,
        pattern=pattern,
        attr=attr,
    )
