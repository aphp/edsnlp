from edsnlp.core import registry
from edsnlp.utils.deprecation import deprecated_factory

from . import patterns
from .accents import AccentsConverter

DEFAULT_CONFIG = dict(
    accents=patterns.accents,
)

create_component = AccentsConverter
create_component = deprecated_factory(
    "accents",
    "eds.accents",
    assigns=["token.norm"],
)(create_component)
create_component = registry.factory.register(
    "eds.accents",
    assigns=["token.norm"],
)(create_component)
