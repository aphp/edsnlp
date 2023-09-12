from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .negation import NegationQualifier

DEFAULT_CONFIG = dict(
    pseudo=None,
    preceding=None,
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
create_component = Language.factory(
    "eds.negation",
    assigns=["span._.negation"],
)(create_component)
