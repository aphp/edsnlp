from edsnlp.core import registry

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

create_component = registry.factory.register(
    "eds.reported_speech",
    assigns=["span._.reported_speech"],
    deprecated=[
        "reported_speech",
        "rspeech",
    ],
)(ReportedSpeechQualifier)
