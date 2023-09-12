from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .matcher import GenericMatcher

DEFAULT_CONFIG = dict(
    terms=None,
    regex=None,
    attr="TEXT",
    ignore_excluded=False,
    ignore_space_tokens=False,
    term_matcher="exact",
    term_matcher_config={},
    span_setter={"ents": True},
)

create_component = GenericMatcher
create_component = deprecated_factory(
    "matcher",
    "eds.matcher",
    assigns=["doc.ents", "doc.spans"],
)(create_component)
create_component = Language.factory(
    "eds.matcher",
    assigns=["doc.ents", "doc.spans"],
)(create_component)
