from typing import List, Optional

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from . import Dates

DEFAULT_CONFIG = dict(
    absolute=None,
    relative=None,
    false_positive=None,
    on_ents_only=False,
    attr="LOWER",
)


@deprecated_factory("dates", "eds.dates", default_config=DEFAULT_CONFIG)
@Language.factory("eds.dates", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    absolute: Optional[List[str]],
    relative: Optional[List[str]],
    false_positive: Optional[List[str]],
    on_ents_only: bool,
    attr: str,
):
    return Dates(
        nlp,
        absolute=absolute,
        relative=relative,
        false_positive=false_positive,
        on_ents_only=on_ents_only,
        attr=attr,
    )
