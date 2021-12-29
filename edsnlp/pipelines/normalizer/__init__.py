from functools import lru_cache
from typing import Union

from spacy.tokens import Doc, Span, Token

from .accents import Accents
from .pollution import Pollution
from .quotes import Quotes

if not Token.has_extension("excluded"):
    Token.set_extension("excluded", default=False)

if not Token.has_extension("excluded_or_space"):
    Token.set_extension(
        "excluded_or_space", getter=lambda t: t.is_space or t._.excluded
    )
