from edsnlp.core import registry

from .global_health_status import GlobalHealthStatusMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="global_health_status",
    label="global_health_status",
    span_setter={"ents": True, "global_health_status": True},
)

create_component = registry.factory.register(
    "eds.global_health_status",
    assigns=["doc.ents", "doc.spans"],
)(GlobalHealthStatusMatcher)
