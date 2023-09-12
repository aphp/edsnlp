from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

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

create_component = deprecated_factory(
    "hypothesis",
    "eds.hypothesis",
    assigns=["span._.hypothesis"],
)(HypothesisQualifier)
create_component = Language.factory(
    "eds.hypothesis",
    assigns=["span._.hypothesis"],
)(create_component)
