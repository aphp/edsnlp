import warnings
from typing import Any, Union

from confit import VisibleDeprecationWarning
from spacy.tokens import Doc, Span, Token


def deprecated_extension(name: str, new_name: str) -> None:
    msg = (
        f'The extension "{name}" is deprecated and will be '
        "removed in a future version. "
        f'Please use "{new_name}" instead.'
    )

    warnings.warn(msg, VisibleDeprecationWarning)


class deprecated_getter_factory:
    def __init__(self, name: str, new_name: str):
        self.name = name
        self.new_name = new_name

    def __call__(self, toklike: Union[Token, Span, Doc]) -> Any:
        n = f"{type(toklike).__name__}._.{self.name}"
        nn = f"{type(toklike).__name__}._.{self.new_name}"

        deprecated_extension(n, nn)

        return getattr(toklike._, self.new_name)
