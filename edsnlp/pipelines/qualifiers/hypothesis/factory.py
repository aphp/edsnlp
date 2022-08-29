from typing import List, Optional

from spacy.language import Language

from edsnlp.pipelines.qualifiers.hypothesis import Hypothesis
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    pseudo=None,
    preceding=None,
    following=None,
    termination=None,
    verbs_hyp=None,
    verbs_eds=None,
    attr="NORM",
    on_ents_only=True,
    within_ents=False,
    explain=False,
)


@deprecated_factory(
    "hypothesis",
    "eds.hypothesis",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.hypothesis"],
)
@Language.factory(
    "eds.hypothesis",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.hypothesis"],
)
def create_component(
    nlp: Language,
    name: str,
    attr: str,
    pseudo: Optional[List[str]],
    preceding: Optional[List[str]],
    following: Optional[List[str]],
    termination: Optional[List[str]],
    verbs_eds: Optional[List[str]],
    verbs_hyp: Optional[List[str]],
    on_ents_only: bool,
    within_ents: bool,
    explain: bool,
):
    return Hypothesis(
        nlp=nlp,
        attr=attr,
        pseudo=pseudo,
        preceding=preceding,
        following=following,
        termination=termination,
        verbs_eds=verbs_eds,
        verbs_hyp=verbs_hyp,
        on_ents_only=on_ents_only,
        within_ents=within_ents,
        explain=explain,
    )
