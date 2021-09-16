from typing import Dict, Any, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.negation import Negation, terms
from edsnlp.pipelines.terminations import termination


negation_default_config = dict(
    pseudo=terms.pseudo,
    preceding=terms.preceding,
    following=terms.following,
    termination=termination,
    verbs=terms.verbs,
)


@Language.factory("negation", default_config=negation_default_config)
def create_component(
    nlp: Language,
    name: str,
    pseudo: List[str],
    preceding: List[str],
    following: List[str],
    termination: List[str],
    verbs: List[str],
    fuzzy: bool = False,
    filter_matches: bool = False,
    attr: str = "LOWER",
    on_ents_only: bool = True,
    regex: Optional[Dict[str, Union[List[str], str]]] = None,
    fuzzy_kwargs: Optional[Dict[str, Any]] = None,
):
    return Negation(
        nlp,
        pseudo=pseudo,
        preceding=preceding,
        following=following,
        termination=termination,
        verbs=verbs,
        fuzzy=fuzzy,
        filter_matches=filter_matches,
        attr=attr,
        on_ents_only=on_ents_only,
        regex=regex,
        fuzzy_kwargs=fuzzy_kwargs,
    )
