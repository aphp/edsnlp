from edsnlp.core import registry

from .dementia import DementiaMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="dementia",
    span_setter={"ents": True, "dementia": True},
)

create_component = registry.factory.register(
    "eds.dementia",
    assigns=["doc.ents", "doc.spans"],
)(DementiaMatcher)
