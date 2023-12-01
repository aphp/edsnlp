from typing import Any, Callable, Dict, Optional, Union

from decorator import decorator
from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span, Token


def deprecated_extension(name: str, new_name: str) -> None:
    msg = (
        f'The extension "{name}" is deprecated and will be '
        "removed in a future version. "
        f'Please use "{new_name}" instead.'
    )

    logger.warning(msg)


def deprecated_getter_factory(name: str, new_name: str) -> Callable:
    def getter(toklike: Union[Token, Span, Doc]) -> Any:

        n = f"{type(toklike).__name__}._.{name}"
        nn = f"{type(toklike).__name__}._.{new_name}"

        deprecated_extension(n, nn)

        return getattr(toklike._, new_name)

    return getter


def deprecation(name: str, new_name: Optional[str] = None):

    new_name = new_name or f"eds.{name}"

    msg = (
        f'Calling "{name}" directly is deprecated and '
        "will be removed in a future version. "
        f'Please use "{new_name}" instead.'
    )

    logger.warning(msg)


def deprecated_factory(
    name: str,
    new_name: Optional[str] = None,
    default_config: Optional[Dict[str, Any]] = None,
    func: Optional[Callable] = None,
    **kwargs,
) -> Callable:
    """
    Execute the Language.factory method on a modified factory function.
    The modification adds a deprecation warning.

    Parameters
    ----------
    name : str
        The deprecated name for the pipeline
    new_name : Optional[str], optional
        The new name for the pipeline, which should be used, by default None
    default_config : Optional[Dict[str, Any]], optional
        The configuration that should be passed to Language.factory, by default None
    func : Optional[Callable], optional
        The function to decorate, by default None

    Returns
    -------
    Callable
    """

    if default_config is None:
        default_config = dict()

    wrapper = Language.factory(name, default_config=default_config, **kwargs)

    def wrap(factory):

        # Define decorator
        # We use micheles' decorator package to keep the same signature
        # See https://github.com/micheles/decorator/
        @decorator
        def decorate(
            f,
            *args,
            **kwargs,
        ):
            deprecation(name, new_name)
            return f(
                *args,
                **kwargs,
            )

        decorated = decorate(factory)

        wrapper(decorated)

        return factory

    if func is not None:
        return wrap(func)

    return wrap
