from edsnlp.core import registry

from .patterns import default_patterns
from .psychological_status import PsychologicalStatusMatcher

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="psychological_status",
    label="psychological_status",
    span_setter={"ents": True, "psychological_status": True},
)

create_component = registry.factory.register(
    "eds.psychological_status",
    assigns=["doc.ents", "doc.spans"],
)(PsychologicalStatusMatcher)
