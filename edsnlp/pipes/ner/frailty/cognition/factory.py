from edsnlp.core import registry

from .cognition import CognitionMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="cognition",
    label="cognition",
    span_setter={"ents": True, "cognition": True},
)

create_component = registry.factory.register(
    "eds.cognition",
    assigns=["doc.ents", "doc.spans"],
)(CognitionMatcher)
