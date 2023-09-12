from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .reason import ReasonMatcher

DEFAULT_CONFIG = dict(
    reasons=None,
    attr="TEXT",
    use_sections=False,
    ignore_excluded=False,
)

create_component = deprecated_factory(
    "reason",
    "eds.reason",
    assigns=["doc.spans", "doc.ents"],
)(ReasonMatcher)
create_component = Language.factory(
    "eds.reason",
    assigns=["doc.spans", "doc.ents"],
)(create_component)
