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

Binding = Tuple[str, Any]


def _check_path(path: str):
    assert [letter.isalnum() or letter == "_" or letter == "." for letter in path], (
        "The label must be a path of valid python identifier to be used as a getter"
        "in the following template: span.[YOUR_LABEL], such as `label_` or `_.negated"
    )
    if path[0].isalpha() or path[0] == "_":
        return "." + path
    return path


def make_binding_getter(qualifier: Union[str, Binding]):
    """
    Make a qualifier getter

    Parameters
    ----------
    qualifier: Union[str, Binding]
        Either one of the following:
        - a path to a nested attributes of the span, such as "qualifier_" or "_.negated"
        - a tuple of (key, value) equality, such as `("_.date.mode", "PASSED")`

    Returns
    -------
    Callable[[Span], bool]
        The qualifier getter
    """
    if isinstance(qualifier, tuple):
        path, value = qualifier
        path = _check_path(path)
        ctx = {"value": value}
        exec(
            f"def getter(span):\n"
            f"    try:\n"
            f"        return span{path} == value\n"
            f"    except AttributeError:\n"
            f"        return False\n",
            ctx,
            ctx,
        )
        return ctx["getter"]
    else:
        path = _check_path(qualifier)
        ctx = {}
        exec(
            f"def getter(span):\n"
            f"    try:\n"
            f"        return span{path}\n"
            f"    except AttributeError:\n"
            f"        return None\n",
            ctx,
            ctx,
        )
        return ctx["getter"]


def make_binding_setter(binding: Binding):
    """
    Make a qualifier setter

    Parameters
    ----------
    binding: Binding
        A pair of
        - a path to a nested attributes of the span, such as `qualifier_` or `_.negated`
        - a value assignment

    Returns
    -------
    Callable[[Span]]
        The qualifier setter
    """
    path, value = binding
    _check_path(path)
    fn_string = f"""def fn(span): span.{path} = value"""
    loc = {"value": value}
    exec(fn_string, loc, loc)
    return loc["fn"]


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

Qualifiers = Union[SeqStr, Dict[str, SpanFilter]]


def validate_qualifiers(value: Union[SeqStr, Dict[str, SpanFilter]]) -> Qualifiers:
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
                    f"Invalid entry {value} ({type(value)}) for Qualifiers, "
                    f"expected bool/string(s), dict of bool/string(s) or callable"
                )
        return new_value
    else:
        raise TypeError(
            f"Invalid entry {value} ({type(value)}) for SpanSetterArg, "
            f"expected bool/string(s), dict of bool/string(s) or callable"
        )


class QualifiersArg:
    """
    Valid values for the `qualifiers` argument of a component can be :

    - a (span) -> qualifier callable
    - a qualifier name ("_.negated")
    - a list of qualifier names (["_.negated", "_.event"])
    - a dict of qualifier name to True or list of labels, to filter the qualifiers

    Examples
    --------
    - `qualifiers="_.negated"` will use the `negated` extention of the span
    - `qualifiers=["_.negated", "_.past"]` will use the `negated` and `past`
       extensions of the span
    - `qualifiers={"_.negated": True, "_.past": "DATE"}` will use the `negated`
       extension of any span, and the `past` extension of spans with the `DATE` label
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value, config=None) -> Qualifiers:
        return validate_qualifiers(value)


if TYPE_CHECKING:
    QualifiersArg = Union[SeqStr, Dict[str, Union[bool, SeqStr]], Callable]  # noqa: F811
