from typing import List, Optional, Union

from spacy.language import Language

from .adicap import Adicap
from .patterns import adicap_prefix, base_code

DEFAULT_CONFIG = dict(
    pattern=base_code,
    prefix=adicap_prefix,
    attr="TEXT",
    window=500,
)


@Language.factory(
    "eds.adicap",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str = "eds.adicap",
    pattern: Optional[Union[List[str], str]] = base_code,
    prefix: Optional[Union[List[str], str]] = adicap_prefix,
    window: int = 500,
    attr: str = "TEXT",
):
    """
    Create a new component to recognize and normalize ADICAP codes in documents.

    Parameters
    ----------
    nlp: Language
        spaCy `Language` object.
    name: str
        The name of the pipe
    pattern: Optional[Union[List[str], str]]
        The regex pattern to use for matching ADICAP codes
    prefix: Optional[Union[List[str], str]]
        The regex pattern to use for matching the prefix before ADICAP codes
    window: int
        Number of tokens to look for prefix. It will never go further the start of
        the sentence
    attr: str
        Attribute to match on, eg `TEXT`, `NORM`, etc.
    """

    return Adicap(
        nlp,
        name=name,
        pattern=pattern,
        attr=attr,
        prefix=prefix,
        window=window,
    )
