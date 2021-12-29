from typing import Dict, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.hypothesis import Hypothesis, terms
from edsnlp.pipelines.terminations import termination

hypothesis_default_config = dict(
    pseudo=terms.pseudo,
    confirmation=terms.confirmation,
    preceding=terms.preceding,
    following=terms.following,
    termination=termination,
    verbs_hyp=terms.verbs_hyp,
    verbs_eds=terms.verbs_eds,
)


@Language.factory("hypothesis", default_config=hypothesis_default_config)
def create_component(
    nlp: Language,
    name: str,
    pseudo: List[str],
    confirmation: List[str],
    preceding: List[str],
    following: List[str],
    termination: List[str],
    verbs_hyp: List[str],
    verbs_eds: List[str],
    filter_matches: bool = False,
    attr: str = "LOWER",
    explain: bool = False,
    on_ents_only: bool = True,
    within_ents: bool = False,
    regex: Optional[Dict[str, Union[List[str], str]]] = None,
    ignore_excluded: bool = False,
):
    return Hypothesis(
        nlp,
        pseudo=pseudo,
        confirmation=confirmation,
        preceding=preceding,
        following=following,
        termination=termination,
        verbs_hyp=verbs_hyp,
        verbs_eds=verbs_eds,
        filter_matches=filter_matches,
        attr=attr,
        explain=explain,
        on_ents_only=on_ents_only,
        within_ents=within_ents,
        regex=regex,
        ignore_excluded=ignore_excluded,
    )
