from edsnlp.core import registry

from .frailty import FrailtyMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="frailty",
    label="frailty",
    span_setter={"ents": True, "frailty": True},
)

create_component = registry.factory.register(
    "eds.frailty",
    assigns=["doc.ents", "doc.spans"],
)(FrailtyMatcher)
