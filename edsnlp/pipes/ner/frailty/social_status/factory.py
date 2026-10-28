from edsnlp.core import registry

from .patterns import default_patterns
from .social_status import SocialMatcher

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="social_status",
    label="social_status",
    span_setter={"ents": True, "social_status": True},
)

create_component = registry.factory.register(
    "eds.social_status",
    assigns=["doc.ents", "doc.spans"],
)(SocialMatcher)
