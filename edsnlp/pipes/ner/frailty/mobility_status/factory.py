from edsnlp.core import registry

from .mobility_status import MobilityMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="mobility_status",
    label="mobility_status",
    span_setter={"ents": True, "mobility_status": True},
)

create_component = registry.factory.register(
    "eds.mobility_status",
    assigns=["doc.ents", "doc.spans"],
)(MobilityMatcher)
