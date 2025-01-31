import functools
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Generic, List, TypeVar, Union

import pydantic
from confit.errors import patch_errors

T = TypeVar("T")

if pydantic.VERSION >= "2":
    from pydantic_core import core_schema


class Validated:
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(v)
                for v in cls.__get_validators__()
            ]
        )


class MetaAsList(type):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        cls.type_ = Any

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, item):
        new_type = MetaAsList(self.__name__, (self,), {})
        new_type.type_ = item
        return new_type

    def validate(cls, value, config=None):
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            value = [value]
        try:
            return cast(List[cls.type_], value)
        except pydantic.ValidationError as e:
            e = patch_errors(e, drop_names=("__root__",))
            e.model = cls
            raise e

    def __get_validators__(cls):
        yield cls.validate

    def __get_pydantic_core_schema__(cls, source, handler):
        return core_schema.no_info_plain_validator_function(cls.validate)


class AsList(Generic[T], metaclass=MetaAsList):
    pass


if pydantic.VERSION < "2":

    def cast(type_, obj):
        class Model(pydantic.BaseModel):
            __root__: type_

            class Config:
                arbitrary_types_allowed = True

        return Model(__root__=obj).__root__
else:
    from dataclasses import is_dataclass

    from pydantic import BaseModel
    from pydantic.type_adapter import ConfigDict, TypeAdapter
    from typing_extensions import is_typeddict

    @lru_cache(maxsize=32)
    def make_type_adapter(type_):
        config = None

        if not (
            (isinstance(type_, type) and issubclass(type_, BaseModel))
            or is_dataclass(type_)
            or is_typeddict(type_)
        ):
            config = ConfigDict(arbitrary_types_allowed=True)
        return TypeAdapter(type_, config=config)

    def cast(type_, obj):
        return make_type_adapter(type_).validate_python(obj)


if TYPE_CHECKING:
    AsList = Union[T, List[T]]  # noqa: F811
