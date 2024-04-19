import pytest
from confit import validate_arguments
from confit.errors import ConfitValidationError

from edsnlp.utils.bindings import BINDING_GETTERS, BINDING_SETTERS, AttributesArg


def test_qualifier_validation():
    @validate_arguments
    def fn(arg: AttributesArg):
        return arg

    assert fn("_.negated") == {"_.negated": True}
    assert fn(["_.negated", "_.event"]) == {"_.negated": True, "_.event": True}
    assert fn({"_.negated": True, "_.event": "DATE"}) == {
        "_.negated": True,
        "_.event": ["DATE"],
    }

    callback = lambda x: x  # noqa: E731

    assert fn(callback) is callback

    with pytest.raises(ConfitValidationError):
        fn(1)

    with pytest.raises(ConfitValidationError):
        fn({"_.negated": 1})


def test_bindings():
    class custom:
        def __init__(self, value):
            self.value = value

    obj = custom([custom(1), custom(2)])
    assert BINDING_GETTERS["value[0].value"](obj) == 1
    assert BINDING_GETTERS[("value[0].value", 1)](obj) is True
    BINDING_SETTERS[("value[1].value", 3)](obj)
    assert obj.value[1].value == 3
