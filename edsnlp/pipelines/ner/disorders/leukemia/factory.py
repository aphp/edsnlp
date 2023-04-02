from edsnlp.core import registry

from .leukemia import LeukemiaMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="leukemia",
    span_setter={"ents": True, "leukemia": True},
)

create_component = registry.factory.register(
    "eds.leukemia",
    assigns=["doc.ents", "doc.spans"],
)(LeukemiaMatcher)
