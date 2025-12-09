from edsnlp.core import registry

from .autonomy import AutonomyMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="autonomy",
    label="autonomy",
    span_setter={"ents": True, "autonomy": True},
)

create_component = registry.factory.register(
    "eds.autonomy",
    assigns=["doc.ents", "doc.spans"],
)(AutonomyMatcher)
