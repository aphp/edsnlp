from edsnlp.core import registry

from .patterns import default_patterns
from .social import SocialMatcher

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="social",
    label="social",
    span_setter={"ents": True, "social": True},
)

create_component = registry.factory.register(
    "eds.social",
    assigns=["doc.ents", "doc.spans"],
)(SocialMatcher)
