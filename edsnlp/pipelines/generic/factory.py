from typing import Dict, Any, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.generic import GenericMatcher

# noinspection PyUnusedLocal
@Language.factory("matcher")
def create_component(
    nlp: Language,
    name: str,
    terms: Optional[Dict[str, Union[str, List[str]]]] = None,
    attr: Union[str, Dict[str, str]] = "TEXT",
    regex: Optional[Dict[str, Union[str, List[str]]]] = None,
    fuzzy: bool = False,
    fuzzy_kwargs: Optional[Dict[str, Any]] = None,
    filter_matches: bool = True,
    on_ents_only: bool = False,
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
        fuzzy=fuzzy,
        fuzzy_kwargs=fuzzy_kwargs,
        filter_matches=filter_matches,
        on_ents_only=on_ents_only,
    )
