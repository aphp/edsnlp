from typing import List, Optional, Union

from spacy.language import Language

from .adicap import Adicap

DEFAULT_CONFIG = dict(pattern=None, prefix=None, attr="TEXT", window=500)


@Language.factory(
    "eds.adicap",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    pattern: Optional[Union[List[str], str]],
    prefix: Optional[Union[List[str], str]],
    window: int,
    attr: str,
):
    return Adicap(nlp, pattern=pattern, attr=attr, prefix=prefix, window=window)
