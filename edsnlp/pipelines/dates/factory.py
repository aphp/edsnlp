from typing import List, Union

from spacy.language import Language

from . import Dates, patterns

default_config = dict(
    no_year=patterns.no_year_pattern,
    year_only=patterns.full_year_pattern,
    no_day=patterns.no_day_pattern,
    absolute=patterns.absolute_date_pattern,
    relative=patterns.relative_date_pattern,
    full=patterns.full_date_pattern,
    current=patterns.current_pattern,
    false_positive=patterns.false_positive_pattern,
)


# noinspection PyUnusedLocal
@Language.factory("dates", default_config=default_config)
def create_component(
    nlp: Language,
    name: str,
    no_year: Union[List[str], str],
    year_only: Union[List[str], str],
    no_day: Union[List[str], str],
    absolute: Union[List[str], str],
    full: Union[List[str], str],
    relative: Union[List[str], str],
    current: Union[List[str], str],
    false_positive: Union[List[str], str],
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
    )
