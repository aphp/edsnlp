import warnings
from typing import Any, Callable, Union

from confit import VisibleDeprecationWarning
from spacy.tokens import Doc, Span, Token


def deprecated_extension(name: str, new_name: str) -> None:
    msg = (
        f'The extension "{name}" is deprecated and will be '
        "removed in a future version. "
        f'Please use "{new_name}" instead.'
    )

    warnings.warn(msg, VisibleDeprecationWarning)


def deprecated_getter_factory(name: str, new_name: str) -> Callable:
    def getter(toklike: Union[Token, Span, Doc]) -> Any:
        n = f"{type(toklike).__name__}._.{name}"
        nn = f"{type(toklike).__name__}._.{new_name}"

        deprecated_extension(n, nn)

        return getattr(toklike._, new_name)

    return getter
