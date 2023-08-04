from typing import List, Optional, Union

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .tnm import TNM

DEFAULT_CONFIG = dict(
    pattern=None,
    attr="TEXT",
)


@deprecated_factory(
    "eds.TNM",
    "eds.tnm",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
@Language.factory(
    "eds.tnm",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    pattern: Optional[Union[List[str], str]],
    attr: str,
):
    return TNM(
        nlp,
        pattern=pattern,
        attr=attr,
    )
