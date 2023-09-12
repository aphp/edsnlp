from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .reported_speech import ReportedSpeechQualifier

DEFAULT_CONFIG = dict(
    pseudo=None,
    preceding=None,
    following=None,
    quotation=None,
    verbs=None,
    attr="NORM",
    span_getter=None,
    on_ents_only=True,
    within_ents=False,
    explain=False,
)

create_component = ReportedSpeechQualifier
create_component = deprecated_factory(
    "reported_speech",
    "eds.reported_speech",
    assigns=["span._.reported_speech"],
)(create_component)
create_component = deprecated_factory(
    "rspeech",
    "eds.reported_speech",
    assigns=["span._.reported_speech"],
)(create_component)
create_component = Language.factory(
    "eds.reported_speech",
    assigns=["span._.reported_speech"],
)(create_component)
