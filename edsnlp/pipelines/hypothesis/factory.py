from typing import Dict, Any, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.hypothesis import Hypothesis, terms


hypothesis_default_config = dict(
    pseudo=terms.pseudo,
    confirmation=terms.confirmation,
    preceding=terms.preceding,
    following=terms.following,
    verbs_hyp=terms.verbs_hyp,
    verbs_eds=terms.verbs_eds,
    on_ents_only=True,
)


@Language.factory("hypothesis", default_config=hypothesis_default_config)
def create_hypothesis_component(
    nlp: Language,
    name: str,
    pseudo: List[str],
    confirmation: List[str],
    preceding: List[str],
    following: List[str],
    verbs_hyp: List[str],
    verbs_eds: List[str],
    fuzzy: bool = False,
    filter_matches: bool = False,
    annotation_scheme: str = "all",
    attr: str = "LOWER",
    on_ents_only: bool = True,
    regex: Optional[Dict[str, Union[List[str], str]]] = None,
    fuzzy_kwargs: Optional[Dict[str, Any]] = None,
):
    return Hypothesis(
        nlp,
        pseudo=pseudo,
        confirmation=confirmation,
        preceding=preceding,
        following=following,
        verbs_hyp=verbs_hyp,
        verbs_eds=verbs_eds,
        fuzzy=fuzzy,
        filter_matches=filter_matches,
        annotation_scheme=annotation_scheme,
        attr=attr,
        on_ents_only=on_ents_only,
        regex=regex,
        fuzzy_kwargs=fuzzy_kwargs,
    )
