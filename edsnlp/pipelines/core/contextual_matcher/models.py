import re
from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, Extra, validator

from edsnlp.matchers.utils import ListOrStr

Flags = Union[re.RegexFlag, int]
Window = Union[
    Tuple[int, int],
    List[int],
    int,
]


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
    regex_flags: Optional[Flags] = None

    @validator("regex")
    def exclude_regex_validation(cls, v):
        if type(v) == str:
            v = [v]
        return v

    _normalize_window = validator("window", allow_reuse=True)(normalize_window)


class ExcludeModel(BaseModel, extra=Extra.forbid):

    __root__: Union[
        List[SingleExcludeModel],
        SingleExcludeModel,
    ]

    @validator("__root__", pre=True)
    def item_to_list(cls, v):
        if not isinstance(v, list):
            return [v]
        return v


class SingleAssignModel(BaseModel):
    name: str
    regex: str
    window: Window
    regex_flags: Optional[Flags] = None
    replace_entity: bool = False
    reduce_mode: Optional[str] = None

    @validator("regex")
    def check_single_regex_group(cls, pat):
        compiled_pat = re.compile(pat)
        n_groups = compiled_pat.groups
        assert n_groups == 1, (
            "The pattern {pat} should have only one capturing group, not {n_groups}"
        ).format(
            pat=pat,
            n_groups=n_groups,
        )

        return pat

    _normalize_window = validator("window", allow_reuse=True)(normalize_window)


class AssignModel(BaseModel, extra=Extra.forbid):

    __root__: Union[
        List[SingleAssignModel],
        SingleAssignModel,
    ]

    @validator("__root__", pre=True)
    def item_to_list(cls, v):
        if not isinstance(v, list):
            return [v]
        return v

    @validator("__root__")
    def name_uniqueness(cls, v):
        names = [item.name for item in v]
        assert len(names) == len(set(names)), "Each `name` field should be unique"
        return v

    @validator("__root__")
    def replace_uniqueness(cls, v):
        replace = [item for item in v if item.replace_entity]
        assert (
            len(replace) <= 1
        ), "Only 1 assign element can be set with `replace_entity=True`"
        return v


class SingleConfig(BaseModel, extra=Extra.forbid):

    source: str
    terms: ListOrStr = []
    regex: ListOrStr = []
    regex_attr: Optional[str] = None
    regex_flags: Union[re.RegexFlag, int] = None
    exclude: Optional[ExcludeModel] = []
    assign: Optional[AssignModel] = []


class FullConfig(BaseModel, extra=Extra.forbid):

    __root__: Union[
        List[SingleConfig],
        SingleConfig,
    ]

    @validator("__root__", pre=True)
    def pattern_to_list(cls, v):
        if not isinstance(v, list):
            return [v]
        return v

    @validator("__root__", pre=True)
    def source_uniqueness(cls, v):
        sources = [item["source"] for item in v]
        assert len(sources) == len(set(sources)), "Each `source` field should be unique"
        return v
