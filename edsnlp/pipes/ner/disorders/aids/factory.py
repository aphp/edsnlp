from edsnlp import registry

from .aids import AIDSMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="aids",
    span_setter={"ents": True, "aids": True},
)

create_component = registry.factory.register(
    "eds.aids",
    assigns=["doc.ents", "doc.spans"],
    deprecated=["eds.AIDS"],
)(AIDSMatcher)
