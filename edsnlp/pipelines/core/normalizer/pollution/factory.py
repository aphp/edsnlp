from typing import Dict, List, Optional, Union

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from . import Pollution

DEFAULT_CONFIG = dict(
    pollution=None,
)


@deprecated_factory("pollution", "eds.pollution", default_config=DEFAULT_CONFIG)
@Language.factory("eds.pollution", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    pollution: Optional[Dict[str, Union[str, List[str]]]],
):
    return Pollution(
        nlp,
        pollution=pollution,
    )
