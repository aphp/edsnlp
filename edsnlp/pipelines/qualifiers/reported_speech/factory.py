from typing import List, Optional

from spacy.language import Language

from edsnlp.pipelines.qualifiers.reported_speech import ReportedSpeech
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    pseudo=None,
    preceding=None,
    following=None,
    quotation=None,
    verbs=None,
    attr="NORM",
    on_ents_only=True,
    within_ents=False,
    explain=False,
)


@deprecated_factory(
    "rspeech",
    "eds.reported_speech",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.reported_speech"],
)
@deprecated_factory(
    "reported_speech",
    "eds.reported_speech",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.reported_speech"],
)
@Language.factory(
    "eds.reported_speech",
    default_config=DEFAULT_CONFIG,
    assigns=["span._.reported_speech"],
)
def create_component(
    nlp: Language,
    name: str,
    attr: str,
    pseudo: Optional[List[str]],
    preceding: Optional[List[str]],
    following: Optional[List[str]],
    quotation: Optional[List[str]],
    verbs: Optional[List[str]],
    on_ents_only: bool,
    within_ents: bool,
    explain: bool,
):
    return ReportedSpeech(
        nlp=nlp,
        attr=attr,
        pseudo=pseudo,
        preceding=preceding,
        following=following,
        quotation=quotation,
        verbs=verbs,
        on_ents_only=on_ents_only,
        within_ents=within_ents,
        explain=explain,
    )
