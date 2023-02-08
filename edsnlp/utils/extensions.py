import functools
from typing import Any, List


def rgetattr(obj: Any, attr: str, *args: List[Any]) -> Any:
    """
    Get attribute recursively

    Parameters
    ----------
    obj : Any
        An object
    attr : str
        The name of the attribute to get. Can contain dots.
    """

    def _getattr(obj, attr):
        return None if obj is None else getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj: Any, attr: str, value: Any):
    """
    Get attribute recursively.
    For instance, if `attr=a.b.c`, then under the hood,
    setattr(obj.a.b, "c", value) is executed

    Parameters
    ----------
    obj : Any
        An object
    attr : str
        The name of the attribute to set. Can contain dots.
    value: Any
        The value to set
    """
    splitted = attr.split(".", maxsplit=1)
    last = splitted.pop(-1)
    attr_obj = rgetattr(obj, ".".join(splitted)) if splitted else obj

    setattr(attr_obj, last, value)
