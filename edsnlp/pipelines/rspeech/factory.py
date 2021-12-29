from typing import List

from spacy.language import Language

from edsnlp.pipelines.rspeech import ReportedSpeech, terms

rspeech_default_config = dict(
    preceding=terms.preceding,
    following=terms.following,
    verbs=terms.verbs,
    quotation=terms.quotation,
    ignore_excluded=False,
)


@Language.factory("rspeech", default_config=rspeech_default_config)
def create_component(
    nlp: Language,
    name: str,
    quotation: str,
    preceding: List[str],
    following: List[str],
    verbs: List[str],
    filter_matches: bool = False,
    attr: str = "LOWER",
    on_ents_only: bool = True,
    within_ents: bool = False,
    explain: bool = False,
    ignore_excluded: bool = False,
):
    return ReportedSpeech(
        nlp,
        quotation=quotation,
        preceding=preceding,
        following=following,
        verbs=verbs,
        filter_matches=filter_matches,
        attr=attr,
        on_ents_only=on_ents_only,
        within_ents=within_ents,
        explain=explain,
        ignore_excluded=ignore_excluded,
    )
