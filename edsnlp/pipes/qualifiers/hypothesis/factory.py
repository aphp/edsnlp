from edsnlp.core import registry

from .hypothesis import HypothesisQualifier

DEFAULT_CONFIG = dict(
    pseudo=None,
    preceding=None,
    following=None,
    verbs_eds=None,
    verbs_hyp=None,
    termination=None,
    attr="NORM",
    span_getter=None,
    on_ents_only=True,
    within_ents=False,
    explain=False,
)

create_component = registry.factory.register(
    "eds.hypothesis",
    assigns=["span._.hypothesis"],
    deprecated=["hypothesis"],
)(HypothesisQualifier)
