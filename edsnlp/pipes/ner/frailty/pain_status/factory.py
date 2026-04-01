from edsnlp.core import registry

from .pain_status import PainMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="pain_status",
    label="pain_status",
    span_setter={"ents": True, "pain_status": True},
)

create_component = registry.factory.register(
    "eds.pain_status",
    assigns=["doc.ents", "doc.spans"],
)(PainMatcher)
