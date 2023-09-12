from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

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

create_component = TerminologyMatcher
create_component = deprecated_factory(
    "terminology",
    "eds.terminology",
    assigns=["doc.ents", "doc.spans"],
)(create_component)
create_component = Language.factory(
    "eds.terminology",
    assigns=["doc.ents", "doc.spans"],
)(create_component)
