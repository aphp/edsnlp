from edsnlp.core import registry

from .terminology import TerminologyMatcher

DEFAULT_CONFIG = dict(
    terms=None,
    regex=None,
    attr="TEXT",
    ignore_excluded=False,
    ignore_space_tokens=False,
    term_matcher="exact",
    term_matcher_config=None,
    span_setter={"ents": True},
)

create_component = registry.factory.register(
    "eds.terminology",
    assigns=["doc.ents", "doc.spans"],
    deprecated=["terminology"],
)(TerminologyMatcher)
