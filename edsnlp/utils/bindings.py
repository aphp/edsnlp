from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Tuple,
    TypeVar,
    Union,
)

from edsnlp.utils.span_getters import SeqStr, SpanFilter
from edsnlp.utils.typing import Validated

Binding = Tuple[str, Any]


def _check_path(path: str):
    assert [letter.isalnum() or letter == "_" or letter == "." for letter in path], (
        "The label must be a path of valid python identifier to be used as a getter"
        "in the following template: span.[YOUR_LABEL], such as `label_` or `_.negated"
    )
    parts = path.split(".")
    new_path = "span"
    for part in parts:
        if " " in part:
            new_path = "getattr(" + new_path + f", {part!r})"
        elif len(part) > 0:
            new_path += "." + part
    return new_path


def make_binding_getter(attribute: Union[str, Binding]):
    """
    Make a attribute getter

    Parameters
    ----------
    attribute: Union[str, Binding]
        Either one of the following:
        - a path to a nested attributes of the span, such as "attribute_" or "_.negated"
        - a tuple of (key, value) equality, such as `("_.date.mode", "PASSED")`

    Returns
    -------
    Callable[[Span], bool]
        The attribute getter
    """
    if isinstance(attribute, tuple):
        path, value = attribute
        path = _check_path(path)
        ctx = {"value": value}
        exec(
            f"def getter(span):\n"
            f"    try:\n"
            f"        return {path} == value\n"
            f"    except AttributeError:\n"
            f"        return False\n",
            ctx,
            ctx,
        )
        return ctx["getter"]
    else:
        path = _check_path(attribute)
        ctx = {}
        exec(
            f"def getter(span):\n"
            f"    try:\n"
            f"        return {path}\n"
            f"    except AttributeError:\n"
            f"        return None\n",
            ctx,
            ctx,
        )
        return ctx["getter"]


def make_binding_setter(binding: Binding):
    """
    Make a attribute setter

    Parameters
    ----------
    binding: Binding
        A pair of
        - a path to a nested attributes of the span, such as `attribute_` or `_.negated`
        - a value assignment

    Returns
    -------
    Callable[[Span]]
        The attribute setter
    """
    if isinstance(binding, tuple):
        path, value = binding
        path = _check_path(path)
        fn_string = f"""def setter(span): {path} = value"""
        ctx = {"value": value}
        exec(fn_string, ctx, ctx)
    else:
        path = _check_path(binding)
        fn_string = f"""def setter(span, value): {path} = value"""
        ctx = {}
        exec(fn_string, ctx, ctx)
    return ctx["setter"]


K = TypeVar("K")
V = TypeVar("V")


class keydefaultdict(dict):
    def __init__(self, default_factory: Callable[[K], V]):
        super().__init__()
        self.default_factory = default_factory

    def __missing__(self, key: K) -> V:
        ret = self[key] = self.default_factory(key)
        return ret


BINDING_GETTERS = keydefaultdict(make_binding_getter)
BINDING_SETTERS = keydefaultdict(make_binding_setter)

Attributes = Dict[str, SpanFilter]


def validate_attributes(value: Union[SeqStr, Dict[str, SpanFilter]]) -> Attributes:
    if callable(value):
        return value
    if isinstance(value, str):
        return {value: True}
    if isinstance(value, list):
        return {qlf: True for qlf in value}
    elif isinstance(value, dict):
        new_value = {}
        for k, v in value.items():
            if isinstance(v, bool):
                new_value[k] = v
            elif isinstance(v, str):
                new_value[k] = [v]
            elif isinstance(v, (list, tuple)) and all(isinstance(i, str) for i in v):
                new_value[k] = list(v)
            else:
                raise TypeError(
                    f"Invalid entry {value} ({type(value)}) for Attributes, "
                    f"expected bool/string(s), dict of bool/string(s) or callable"
                )
        return new_value
    else:
        raise TypeError(
            f"Invalid entry {value} ({type(value)}) for SpanSetterArg, "
            f"expected bool/string(s), dict of bool/string(s) or callable"
        )


class AttributesArg(Validated):
    """
    Valid values for the `attributes` argument of a component can be :

    - a (span) -> attribute callable
    - a attribute name ("_.negated")
    - a list of attribute names (["_.negated", "_.event"])
    - a dict of attribute name to True or list of labels, to filter the attributes

    Examples
    --------
    - `attributes="_.negated"` will use the `negated` extention of the span
    - `attributes=["_.negated", "_.past"]` will use the `negated` and `past`
       extensions of the span
    - `attributes={"_.negated": True, "_.past": "DATE"}` will use the `negated`
       extension of any span, and the `past` extension of spans with the `DATE` label
    """

    @classmethod
    def validate(cls, value, config=None) -> Attributes:
        return validate_attributes(value)


if TYPE_CHECKING:
    AttributesArg = Union[SeqStr, Dict[str, Union[bool, SeqStr]], Callable]  # noqa: F811
