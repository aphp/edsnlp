from edsnlp.core import registry

from .frailty_mentions import FrailtyMentionsMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="frailty_mentions",
    label="frailty_mentions",
    span_setter={"ents": True, "frailty_mentions": True},
)

create_component = registry.factory.register(
    "eds.frailty_mentions",
    assigns=["doc.ents", "doc.spans"],
)(FrailtyMentionsMatcher)
