from functools import lru_cache
from typing import Union

from spacy.tokens import Doc, Span, Token

from .accents import Accents
from .pollution import Pollution
from .quotes import Quotes

if not Token.has_extension("excluded"):
    Token.set_extension("excluded", default=False)


def excluded_or_space_getter(t):
    return t.is_space or t.tag_ == "EXCLUDED"


if not Token.has_extension("excluded_or_space"):
    Token.set_extension(
        "excluded_or_space",
        getter=excluded_or_space_getter,
    )
