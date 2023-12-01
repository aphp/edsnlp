from edsnlp.core import registry

from .negation import NegationQualifier

DEFAULT_CONFIG = dict(
    pseudo=None,
    preceding=None,
    preceding_regex=None,
    following=None,
    verbs=None,
    termination=None,
    attr="NORM",
    span_getter=None,
    on_ents_only=True,
    within_ents=False,
    explain=False,
)

create_component = registry.factory.register(
    "eds.negation",
    assigns=["span._.negation"],
    deprecated=["negation"],
)(NegationQualifier)
