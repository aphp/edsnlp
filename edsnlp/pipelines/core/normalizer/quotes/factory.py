from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .patterns import quotes_and_apostrophes
from .quotes import QuotesConverter

DEFAULT_CONFIG = dict(
    quotes=quotes_and_apostrophes,
)

create_component = QuotesConverter
create_component = deprecated_factory(
    "quotes",
    "eds.quotes",
    assigns=["token.norm"],
)(create_component)
create_component = Language.factory(
    "eds.quotes",
    assigns=["token.norm"],
)(create_component)
