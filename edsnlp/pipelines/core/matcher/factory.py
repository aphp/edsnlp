from typing import Dict, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.core.matcher import GenericMatcher

default_config = dict(
    terms=None,
    regex=None,
    attr="TEXT",
    filter_matches=True,
    on_ents_only=False,
    ignore_excluded=False,
)


# noinspection PyUnusedLocal
@Language.factory("eds.matcher", default_config=default_config)
def create_component(
    nlp: Language,
    name: str,
    terms: Optional[Dict[str, Union[str, List[str]]]],
    attr: Union[str, Dict[str, str]],
    regex: Optional[Dict[str, Union[str, List[str]]]],
    filter_matches: bool,
    on_ents_only: bool,
    ignore_excluded: bool,
):
    if terms is None:
        terms = dict()
    if regex is None:
        regex = dict()

    return GenericMatcher(
        nlp,
        terms=terms,
        attr=attr,
        regex=regex,
        filter_matches=filter_matches,
        on_ents_only=on_ents_only,
        ignore_excluded=ignore_excluded,
    )


# noinspection PyUnusedLocal
@Language.factory("matcher", default_config=default_config)
def create_component_leg(
    nlp: Language,
    name: str,
    terms: Optional[Dict[str, Union[str, List[str]]]],
    attr: Union[str, Dict[str, str]],
    regex: Optional[Dict[str, Union[str, List[str]]]],
    filter_matches: bool,
    on_ents_only: bool,
    ignore_excluded: bool,
):
    if terms is None:
        terms = dict()
    if regex is None:
        regex = dict()

    return GenericMatcher(
        nlp,
        terms=terms,
        attr=attr,
        regex=regex,
        filter_matches=filter_matches,
        on_ents_only=on_ents_only,
        ignore_excluded=ignore_excluded,
    )
