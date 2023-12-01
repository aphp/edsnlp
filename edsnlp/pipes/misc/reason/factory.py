from edsnlp.core import registry

from .reason import ReasonMatcher

DEFAULT_CONFIG = dict(
    reasons=None,
    attr="TEXT",
    use_sections=False,
    ignore_excluded=False,
)

create_component = registry.factory.register(
    "eds.reason",
    assigns=["doc.spans", "doc.ents"],
    deprecated=["reason"],
)(ReasonMatcher)
