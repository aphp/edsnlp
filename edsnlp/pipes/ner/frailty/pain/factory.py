from edsnlp.core import registry

from .pain import PainMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="pain",
    label="pain",
    span_setter={"ents": True, "pain": True},
)

create_component = registry.factory.register(
    "eds.pain",
    assigns=["doc.ents", "doc.spans"],
)(PainMatcher)
