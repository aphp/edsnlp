from typing import Dict, Optional

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from . import Pollution

DEFAULT_CONFIG = dict(
    pollution=dict(
        information=True,
        bars=True,
        biology=False,
        doctors=True,
        web=True,
        coding=False,
    ),
)


@deprecated_factory(
    "pollution", "eds.pollution", default_config=DEFAULT_CONFIG, assigns=["token.tag"]
)
@Language.factory("eds.pollution", default_config=DEFAULT_CONFIG, assigns=["token.tag"])
def create_component(
    nlp: Language,
    name: str,
    pollution: Optional[Dict[str, bool]],
):
    return Pollution(
        nlp,
        pollution=pollution,
    )
