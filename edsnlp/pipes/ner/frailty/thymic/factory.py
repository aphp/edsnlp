from edsnlp.core import registry

from .patterns import default_patterns
from .thymic import ThymicMatcher

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="thymic",
    label="thymic",
    span_setter={"ents": True, "thymic": True},
)

create_component = registry.factory.register(
    "eds.thymic",
    assigns=["doc.ents", "doc.spans"],
)(ThymicMatcher)
