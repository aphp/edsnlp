from edsnlp.core import registry

from .patterns import default_patterns
from .sensory import SensoryMatcher

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="sensory",
    label="sensory",
    span_setter={"ents": True, "sensory": True},
)

create_component = registry.factory.register(
    "eds.sensory",
    assigns=["doc.ents", "doc.spans"],
)(SensoryMatcher)
