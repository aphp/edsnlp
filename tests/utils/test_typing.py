import pytest
from confit import validate_arguments
from confit.errors import ConfitValidationError

from edsnlp.utils.typing import AsList


def test_as_list():
    @validate_arguments
    def func(a: AsList[int]):
        return a

    assert func("1") == [1]

    with pytest.raises(ConfitValidationError) as e:
        func("a")

    assert (
        "1 validation error for test_typing.test_as_list.<locals>.func()\n" "-> a.0\n"
    ) in str(e.value)
