from typing import TYPE_CHECKING, Any, Generic, List, TypeVar

import pydantic
from confit.errors import patch_errors

T = TypeVar("T")


class MetaAsList(type):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        cls.item = Any

    def __getitem__(self, item):
        new_type = MetaAsList(self.__name__, (self,), {})
        new_type.item = item
        return new_type

    def validate(cls, value, config=None):
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            value = [value]
        try:
            return pydantic.parse_obj_as(List[cls.item], value)
        except pydantic.ValidationError as e:
            e = patch_errors(e, drop_names=("__root__",))
            e.model = cls
            raise e

    def __get_validators__(cls):
        yield cls.validate


class AsList(Generic[T], metaclass=MetaAsList):
    pass


if TYPE_CHECKING:
    AsList = List[T]  # noqa: F811
