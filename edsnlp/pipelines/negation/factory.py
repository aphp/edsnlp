from typing import Dict, List, Optional, Union

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
    filter_matches: bool = False,
    attr: str = "LOWER",
    on_ents_only: bool = True,
    within_ents: bool = False,
    regex: Optional[Dict[str, Union[List[str], str]]] = None,
    explain: bool = False,
    ignore_excluded: bool = False,
):
    return Negation(
        nlp,
        pseudo=pseudo,
        preceding=preceding,
        following=following,
        termination=termination,
        verbs=verbs,
        filter_matches=filter_matches,
        attr=attr,
        on_ents_only=on_ents_only,
        within_ents=within_ents,
        regex=regex,
        explain=explain,
        ignore_excluded=ignore_excluded,
    )
