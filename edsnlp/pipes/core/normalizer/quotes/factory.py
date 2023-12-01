from edsnlp.core import registry

from .patterns import quotes_and_apostrophes
from .quotes import QuotesConverter

DEFAULT_CONFIG = dict(
    quotes=quotes_and_apostrophes,
)

create_component = registry.factory.register(
    "eds.quotes",
    assigns=["token.norm"],
    deprecated=["quotes"],
)(QuotesConverter)
