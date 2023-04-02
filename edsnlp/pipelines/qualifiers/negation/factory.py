from edsnlp.core import registry
from edsnlp.utils.deprecation import deprecated_factory

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

create_component = deprecated_factory(
    "negation",
    "eds.negation",
    assigns=["span._.negation"],
)(NegationQualifier)
create_component = registry.factory.register(
    "eds.negation",
    assigns=["span._.negation"],
)(create_component)
