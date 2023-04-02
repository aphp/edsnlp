from edsnlp.core import registry

from .alcohol import AlcoholMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="alcohol",
    span_setter={"ents": True, "alcohol": True},
)

create_component = registry.factory.register(
    "eds.alcohol",
    assigns=["doc.ents", "doc.spans"],
)(AlcoholMatcher)
