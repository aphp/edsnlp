from typing import Dict, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.core.matcher import GenericMatcher

from . import patterns

DEFAULT_CONFIG = dict(
    covid=None,
    attr="LOWER",
    ignore_excluded=False,
)


@Language.factory("eds.covid", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    covid: Optional[Dict[str, Union[str, List[str]]]],
    attr: Union[str, Dict[str, str]],
    ignore_excluded: bool,
):

    if covid is None:
        covid = [patterns.pattern]

    return GenericMatcher(
        nlp,
        terms=None,
        regex=dict(covid=covid),
        attr=attr,
        ignore_excluded=ignore_excluded,
    )
