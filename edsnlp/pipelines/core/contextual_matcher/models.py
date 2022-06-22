import re
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Extra, root_validator, validator

from edsnlp.matchers.utils import ListOrStr

Flags = Union[re.RegexFlag, int]


class SidedExcludeModel(BaseModel):
    regex: ListOrStr = []
    regex_flags: Optional[Flags] = None
    window: int = 10

    @validator("regex")
    def exclude_regex_validation(cls, v):
        if type(v) == str:
            v = [v]
        return v


class ExcludeModel(BaseModel):
    after: SidedExcludeModel = SidedExcludeModel()
    before: SidedExcludeModel = SidedExcludeModel()


class SidedAssignModel(BaseModel):
    regex: Dict[str, str] = dict()
    regex_flags: Optional[Flags] = None
    window: int = 10
    expand_entity: bool = False

    @validator("regex")
    def check_single_regex_group(cls, v):
        for pat_name, pat in v.items():
            compiled_pat = re.compile(pat)
            n_groups = compiled_pat.groups
            assert n_groups == 1, (
                "The pattern for {pat_name} ({pat}) should have only one"
                "capturing group, not {n_groups}"
            ).format(
                pat_name=pat_name,
                pat=pat,
                n_groups=n_groups,
            )

        return v


class AssignModel(BaseModel):
    after: SidedAssignModel = SidedAssignModel()
    before: SidedAssignModel = SidedAssignModel()

    @root_validator()
    def check_keys_uniqueness(cls, values):

        common_keys = set(values.get("before").regex.keys()) & set(
            values.get("after").regex.keys()
        )
        assert not common_keys, (
            f"The keys {common_keys} are present in"
            " both 'before' and 'after' assign dictionaries"
        )
        return values


class SingleConfig(BaseModel, extra=Extra.forbid):

    source: str
    terms: ListOrStr = []
    regex: ListOrStr = []
    regex_attr: Optional[str] = None
    regex_flags: Union[re.RegexFlag, int] = None
    exclude: Optional[ExcludeModel] = ExcludeModel()
    assign: Optional[AssignModel]


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
