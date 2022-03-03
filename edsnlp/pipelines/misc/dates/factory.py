from typing import List, Optional

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from . import Dates

DEFAULT_CONFIG = dict(
    no_year=None,
    year_only=None,
    no_day=None,
    absolute=None,
    relative=None,
    full=None,
    current=None,
    false_positive=None,
    on_ents_only=False,
    attr="LOWER",
)


@deprecated_factory("dates", "eds.dates", default_config=DEFAULT_CONFIG)
@Language.factory("eds.dates", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    no_year: Optional[List[str]],
    year_only: Optional[List[str]],
    no_day: Optional[List[str]],
    absolute: Optional[List[str]],
    full: Optional[List[str]],
    relative: Optional[List[str]],
    current: Optional[List[str]],
    false_positive: Optional[List[str]],
    on_ents_only: bool,
    attr: str,
):
    return Dates(
        nlp,
        no_year=no_year,
        absolute=absolute,
        relative=relative,
        year_only=year_only,
        no_day=no_day,
        full=full,
        current=current,
        false_positive=false_positive,
        on_ents_only=on_ents_only,
        attr=attr,
    )
