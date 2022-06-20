import re
from typing import Dict, Optional, Union

from pydantic import BaseModel, Extra, validator

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
    expand_entity: bool = True

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


class SingleConfig(BaseModel, extra=Extra.forbid):

    source: str
    terms: ListOrStr = []
    regex: ListOrStr = []
    regex_attr: Optional[str] = None
    regex_flags: Union[re.RegexFlag, int] = None
    exclude: Optional[ExcludeModel] = ExcludeModel()
    assign: Optional[AssignModel]
