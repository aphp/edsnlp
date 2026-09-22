from edsnlp.core import registry

from .cognitive_status import CognitiveStatusMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="cognitive_status",
    label="cognitive_status",
    span_setter={"ents": True, "cognitive_status": True},
)

create_component = registry.factory.register(
    "eds.cognitive_status",
    assigns=["doc.ents", "doc.spans"],
)(CognitiveStatusMatcher)
