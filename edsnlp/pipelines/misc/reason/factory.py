from edsnlp.core import registry
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
create_component = registry.factory.register(
    "eds.reason",
    assigns=["doc.spans", "doc.ents"],
)(create_component)
