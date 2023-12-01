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
