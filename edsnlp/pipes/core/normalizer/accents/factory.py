from edsnlp.core import registry

from . import patterns
from .accents import AccentsConverter

DEFAULT_CONFIG = dict(
    accents=patterns.accents,
)

create_component = registry.factory.register(
    "eds.accents",
    assigns=["token.norm"],
    deprecated=["accents"],
)(AccentsConverter)
