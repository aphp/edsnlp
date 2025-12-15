from edsnlp.core import registry

from .mobility import MobilityMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="mobility",
    label="mobility",
    span_setter={"ents": True, "mobility": True},
)

create_component = registry.factory.register(
    "eds.mobility",
    assigns=["doc.ents", "doc.spans"],
)(MobilityMatcher)
