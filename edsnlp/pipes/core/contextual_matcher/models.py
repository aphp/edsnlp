import re
from typing import List, Optional, Tuple, Union

import pydantic
import regex
from pydantic import BaseModel, Extra

from edsnlp.matchers.utils import ListOrStr
from edsnlp.utils.typing import Validated, cast

Flags = Union[re.RegexFlag, int]
Window = Union[
    Tuple[int, int],
    List[int],
    int,
]

try:
    from pydantic import field_validator

    def validator(x, allow_reuse=True, pre=False):
        return field_validator(x, mode="before" if pre else "after")
except ImportError:
    from pydantic import validator


def normalize_window(cls, v):
    if isinstance(v, list):
        assert (
            len(v) == 2
        ), "`window` should be a tuple/list of two integer, or a single integer"
        v = tuple(v)
    if isinstance(v, int):
        assert v != 0, "The provided `window` should not be 0"
        if v < 0:
            return (v, 0)
        if v > 0:
            return (0, v)
    assert v[0] < v[1], "The provided `window` should contain at least 1 token"
    return v


class AssignDict(dict):
    """
    Custom dictionary that overrides the __setitem__ method
    depending on the reduce_mode
    """

    def __init__(self, reduce_mode: dict):
        super().__init__()
        self.reduce_mode = reduce_mode
        self._setitem_ = self.__setitem_options__()

    def __missing__(self, key):
        return (
            {
                "span": [],
                "value_span": [],
                "value_text": [],
            }
            if self.reduce_mode[key] is None
            else {}
        )

    def __setitem__(self, key, value):
        self._setitem_[self.reduce_mode[key]](key, value)

    def __setitem_options__(self):
        def keep_list(key, value):
            old_values = self.__getitem__(key)
            value["span"] = old_values["span"] + [value["span"]]
            value["value_span"] = old_values["value_span"] + [value["value_span"]]
            value["value_text"] = old_values["value_text"] + [value["value_text"]]

            dict.__setitem__(self, key, value)

        def keep_first(key, value):
            old_values = self.__getitem__(key)
            if (
                old_values.get("span") is None
                or value["span"].start <= old_values["span"].start
            ):
                dict.__setitem__(self, key, value)

        def keep_last(key, value):
            old_values = self.__getitem__(key)
            if (
                old_values.get("span") is None
                or value["span"].start >= old_values["span"].start
            ):
                dict.__setitem__(self, key, value)

        return {
            None: keep_list,
            "keep_first": keep_first,
            "keep_last": keep_last,
        }


class SingleExcludeModel(BaseModel):
    regex: ListOrStr = []
    window: Window
    limit_to_sentence: Optional[bool] = True
    regex_flags: Optional[Flags] = None
    regex_attr: Optional[str] = None

    @validator("regex")
    def exclude_regex_validation(cls, v):
        if isinstance(v, str):
            v = [v]
        return v

    _normalize_window = validator("window", allow_reuse=True)(normalize_window)
    if pydantic.VERSION < "2":
        model_dump = BaseModel.dict


class ExcludeModel(Validated):
    @classmethod
    def validate(cls, v, config=None):
        if not isinstance(v, list):
            v = [v]
        return [cast(SingleExcludeModel, x) for x in v]

    if pydantic.VERSION < "2":
        model_dump = BaseModel.dict


class SingleAssignModel(BaseModel):
    name: str
    regex: str
    window: Window
    limit_to_sentence: Optional[bool] = True
    regex_flags: Optional[Flags] = None
    regex_attr: Optional[str] = None
    replace_entity: bool = False
    reduce_mode: Optional[str] = None

    @validator("regex")
    def check_single_regex_group(cls, pat):
        compiled_pat = regex.compile(
            pat
        )  # Using regex to allow multiple fgroups with same name
        n_groups = compiled_pat.groups
        assert n_groups == 1, (
            "The pattern {pat} should have only one capturing group, not {n_groups}"
        ).format(
            pat=pat,
            n_groups=n_groups,
        )

        return pat

    _normalize_window = validator("window", allow_reuse=True)(normalize_window)
    if pydantic.VERSION < "2":
        model_dump = BaseModel.dict


class AssignModel(Validated):
    @classmethod
    def item_to_list(cls, v, config=None):
        if not isinstance(v, list):
            v = [v]
        return [cast(SingleAssignModel, x) for x in v]

    @classmethod
    def name_uniqueness(cls, v, config=None):
        names = [item.name for item in v]
        assert len(names) == len(set(names)), "Each `name` field should be unique"
        return v

    @classmethod
    def replace_uniqueness(cls, v, config=None):
        replace = [item for item in v if item.replace_entity]
        assert (
            len(replace) <= 1
        ), "Only 1 assign element can be set with `replace_entity=True`"
        return v

    @classmethod
    def __get_validators__(cls):
        yield cls.item_to_list
        yield cls.name_uniqueness
        yield cls.replace_uniqueness

    if pydantic.VERSION < "2":
        model_dump = BaseModel.dict


class SingleConfig(BaseModel, extra=Extra.forbid):
    source: str
    terms: ListOrStr = []
    regex: ListOrStr = []
    regex_attr: Optional[str] = None
    regex_flags: Union[re.RegexFlag, int] = None
    exclude: Optional[ExcludeModel] = []
    assign: Optional[AssignModel] = []
    if pydantic.VERSION < "2":
        model_dump = BaseModel.dict


class FullConfig(Validated):
    @classmethod
    def pattern_to_list(cls, v, config=None):
        if not isinstance(v, list):
            v = [v]
        return [cast(SingleConfig, item) for item in v]

    @classmethod
    def source_uniqueness(cls, v, config=None):
        sources = [item.source for item in v]
        assert len(sources) == len(set(sources)), "Each `source` field should be unique"
        return v

    @classmethod
    def __get_validators__(cls):
        yield cls.pattern_to_list
        yield cls.source_uniqueness

    if pydantic.VERSION < "2":
        model_dump = BaseModel.dict
