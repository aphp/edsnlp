from typing import Any, Dict, List, Optional

from spacy.language import Language

from edsnlp.pipelines.rspeech import ReportedSpeech, terms

rspeech_default_config = dict(
    preceding=terms.preceding,
    following=terms.following,
    verbs=terms.verbs,
    quotation=terms.quotation,
)


@Language.factory("rspeech", default_config=rspeech_default_config)
def create_component(
    nlp: Language,
    name: str,
    quotation: str,
    preceding: List[str],
    following: List[str],
    verbs: List[str],
    fuzzy: bool = False,
    filter_matches: bool = False,
    attr: str = "LOWER",
    on_ents_only: bool = True,
    fuzzy_kwargs: Optional[Dict[str, Any]] = None,
):
    return ReportedSpeech(
        nlp,
        quotation=quotation,
        preceding=preceding,
        following=following,
        verbs=verbs,
        fuzzy=fuzzy,
        filter_matches=filter_matches,
        attr=attr,
        on_ents_only=on_ents_only,
        fuzzy_kwargs=fuzzy_kwargs,
    )
