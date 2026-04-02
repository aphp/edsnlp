from edsnlp.core import registry

from .patterns import default_patterns
from .sensory_status import SensoryMatcher

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="sensory_status",
    label="sensory_status",
    span_setter={"ents": True, "sensory_status": True},
)

create_component = registry.factory.register(
    "eds.sensory_status",
    assigns=["doc.ents", "doc.spans"],
)(SensoryMatcher)
