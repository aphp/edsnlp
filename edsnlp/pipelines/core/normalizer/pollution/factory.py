from typing import Dict, List, Union

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from . import Pollution, patterns

DEFAULT_CONFIG = dict(
    pollution=patterns.pollution,
)


@deprecated_factory("pollution", "eds.pollution", default_config=DEFAULT_CONFIG)
@Language.factory("eds.pollution", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    pollution: Dict[str, Union[str, List[str]]],
):
    return Pollution(
        nlp,
        pollution=pollution,
    )
