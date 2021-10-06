from typing import Dict, Any, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.family import FamilyContext, terms


family_default_config = dict(family=terms.family)


@Language.factory("family", default_config=family_default_config)
def create_component(
    nlp: Language,
    name: str,
    family: List[str],
    fuzzy: bool = False,
    filter_matches: bool = False,
    attr: str = "LOWER",
    explain: bool = True,
    on_ents_only: bool = True,
    regex: Optional[Dict[str, Union[List[str], str]]] = None,
    fuzzy_kwargs: Optional[Dict[str, Any]] = None,
):
    return FamilyContext(
        nlp,
        family=family,
        fuzzy=fuzzy,
        filter_matches=filter_matches,
        attr=attr,
        explain=explain,
        on_ents_only=on_ents_only,
        regex=regex,
        fuzzy_kwargs=fuzzy_kwargs,
    )
