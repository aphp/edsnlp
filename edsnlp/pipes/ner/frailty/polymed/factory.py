from edsnlp.core import registry

from .patterns import default_patterns
from .polymed import PolymedMatcher

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="polymed",
    label="polymed",
    span_setter={"ents": True, "polymed": True},
)

create_component = registry.factory.register(
    "eds.polymed",
    assigns=["doc.ents", "doc.spans"],
)(PolymedMatcher)
