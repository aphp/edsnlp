import re
from typing import TYPE_CHECKING, Any, List, Optional, Union

import regex
from pydantic import BaseModel

from edsnlp.matchers.utils import ListOrStr
from edsnlp.utils.span_getters import (
    ContextWindow,
    SentenceContextWindow,
    SpanGetterArg,
)
from edsnlp.utils.typing import AsList

Flags = Union[re.RegexFlag, int]

try:
    from pydantic import field_validator, model_validator

    def validator(x, allow_reuse=True, pre=False):
        return field_validator(x, mode="before" if pre else "after")

    def root_validator(allow_reuse=True, pre=False):
        return model_validator(mode="before" if pre else "after")


except ImportError:
    from pydantic import root_validator, validator


def validate_window(cls, values):
    if isinstance(values.get("regex"), str):
        values["regex"] = [values["regex"]]
    window = values.get("window")
    if window is None or isinstance(window, (int, tuple, list)):
        values["limit_to_sentence"] = True
    window = values.get("window")
    if window is not None:
        values["window"] = ContextWindow.validate(window)
    if values.get("limit_to_sentence"):
        values["window"] = (
            SentenceContextWindow(0, 0) & values.get("window")
            if window is not None
            else SentenceContextWindow(0, 0)
        )
    return values


class SingleExcludeModel(BaseModel):
    """
    A dictionary to define exclusion rules. Exclusion rules are given as Regexes, and
    if a match is found in the surrounding context of an extraction, the extraction is
    removed. Note that only take a match into account if it is not inside the anchor
    span.

    Parameters
    ----------
    regex : ListOrStr
        A single Regex or a list of Regexes
    regex_attr : Optional[str]
        An attributes to overwrite the given `attr` when matching with Regexes.
    regex_flags : re.RegexFlag
        Regex flags
    span_getter : Optional[SpanGetterArg]
        A span getter to pick the assigned spans from already extracted entities.
    window : Optional[ContextWindow]
        Context window to search for patterns around the anchor. Defaults to "sent" (
        i.e. the sentence of the anchor span).
    """

    span_getter: Optional[SpanGetterArg] = None
    regex: ListOrStr = []
    regex_attr: Optional[str] = None
    regex_flags: Union[re.RegexFlag, int] = None

    limit_to_sentence: Optional[bool] = None
    window: Optional[ContextWindow] = None
    regex_matcher: Optional[Any] = None

    validate_window = root_validator(pre=True, allow_reuse=True)(validate_window)


class SingleIncludeModel(BaseModel):
    """
    A dictionary to define inclusion rules. Inclusion rules are given as Regexes, and
    if a match isn't found in the surrounding context of an extraction, the extraction
    is removed. Note that only take a match into account if it is not inside the anchor
    span.

    Parameters
    ----------
    regex : ListOrStr
        A single Regex or a list of Regexes
    regex_attr : Optional[str]
        An attributes to overwrite the given `attr` when matching with Regexes.
    regex_flags : re.RegexFlag
        Regex flags
    span_getter : Optional[SpanGetterArg]
        A span getter to pick the assigned spans from already extracted entities.
    window : Optional[ContextWindow]
        Context window to search for patterns around the anchor. Defaults to "sent" (
        i.e. the sentence of the anchor span).
    """

    span_getter: Optional[SpanGetterArg] = None
    regex: ListOrStr = []
    regex_attr: Optional[str] = None
    regex_flags: Union[re.RegexFlag, int] = None

    limit_to_sentence: Optional[bool] = None
    window: Optional[ContextWindow] = None

    regex_matcher: Optional[Any] = None

    validate_window = root_validator(pre=True, allow_reuse=True)(validate_window)


class ExcludeModel(AsList[SingleExcludeModel]):
    """
    A list of `SingleExcludeModel` objects. If a single config is passed,
    it will be automatically converted to a list of a single element.
    """


class IncludeModel(AsList[SingleIncludeModel]):
    """
    A list of `SingleIncludeModel` objects. If a single config is passed,
    it will be automatically converted to a list of a single element.
    """


class SingleAssignModel(BaseModel):
    """
    A dictionary to refine the extraction. Similarly to the `exclude` key, you can
    provide a dictionary to use on the context **before** and **after** the extraction.

    Parameters
    ----------
    span_getter : Optional[SpanGetterArg]
        A span getter to pick the assigned spans from already extracted entities
        in the doc.
    regex : ListOrStr
        A single Regex or a list of Regexes
    regex_attr : Optional[str]
        An attributes to overwrite the given `attr` when matching with Regexes.
    regex_flags : re.RegexFlag
        Regex flags
    window : Optional[ContextWindow]
        Context window to search for patterns around the anchor. Defaults to "sent" (
        i.e. the sentence of the anchor span).
    replace_entity : Optional[bool]
        If set to `True`, the match from the corresponding assign key will be used as
        entity, instead of the main match.
        See [this paragraph][replace_entity]
    reduce_mode : Optional[Flags]
        Set how multiple assign matches are handled. See the documentation of the
        [`reduce_mode` parameter][reduce_mode]
    required : Optional[str]
        If set to `True`, the assign key must match for the extraction to be kept. If
        it does not match, the extraction is discarded.
    name : str
        A name (string)
    """

    name: str

    span_getter: Optional[SpanGetterArg] = None
    regex: ListOrStr = []
    regex_attr: Optional[str] = None
    regex_flags: Union[re.RegexFlag, int] = None

    limit_to_sentence: Optional[bool] = None
    window: Optional[ContextWindow] = None
    replace_entity: bool = False
    reduce_mode: Optional[str] = None
    required: Optional[bool] = False

    regex_matcher: Optional[Any] = None

    @validator("regex", allow_reuse=True)
    def check_single_regex_group(cls, pat):
        for single_pat in pat:
            compiled_pat = regex.compile(
                single_pat
            )  # Using regex to allow multiple fgroups with same name
            n_groups = compiled_pat.groups
            assert n_groups == 1, (
                f"The pattern {single_pat} should have exactly one capturing group, "
                f"not {n_groups}"
            )

        return pat

    validate_window = root_validator(pre=True, allow_reuse=True)(validate_window)


class AssignModel(AsList[SingleAssignModel]):
    """
    A list of `SingleAssignModel` objects that should have at most
    one element with `replace_entity=True`. If a single config is passed,
    it will be automatically converted to a list of a single element.
    """

    @classmethod
    def name_uniqueness(cls, v, config=None):
        names = [item.name for item in v]
        assert len(names) == len(set(names)), "Each `name` field should be unique"
        return v

    @classmethod
    def replace_uniqueness(cls, v, config=None):
        replace = [item for item in v if item.replace_entity]
        assert len(replace) <= 1, (
            "Only 1 assign element can be set with `replace_entity=True`"
        )
        return v

    @classmethod
    def __get_validators__(cls):
        yield cls.validate
        yield cls.name_uniqueness
        yield cls.replace_uniqueness


if TYPE_CHECKING:
    ExcludeModel = List[SingleExcludeModel]  # noqa: F811
    IncludeModel = List[SingleIncludeModel]  # noqa: F811
    AssignModel = List[SingleAssignModel]  # noqa: F811


class SingleConfig(BaseModel, extra="forbid"):
    """
    A single configuration for the contextual matcher.

    Parameters
    ----------
    span_getter : Optional[SpanGetterArg]
        A span getter to pick the assigned spans from already extracted entities
        in the doc.
    regex : ListOrStr
        A single Regex or a list of Regexes
    regex_attr : Optional[str]
        An attributes to overwrite the given `attr` when matching with Regexes.
    regex_flags: re.RegexFlag
        Regex flags
    terms : Union[re.RegexFlag, int]
        A single term or a list of terms (for exact matches)
    exclude : AsList[SingleExcludeModel]
        ??? subdoc "One or more exclusion patterns"

            ::: edsnlp.pipes.core.contextual_matcher.models.SingleExcludeModel
                options:
                    only_parameters: "no-header"
    include : AsList[SingleIncludeModel]
        ??? subdoc "One or more inclusion patterns"

            ::: edsnlp.pipes.core.contextual_matcher.models.SingleIncludeModel
                options:
                    only_parameters: "no-header"
    assign : AsList[SingleAssignModel]
        ??? subdoc "One or more assignment patterns"

            ::: edsnlp.pipes.core.contextual_matcher.models.SingleAssignModel
                options:
                    only_parameters: "no-header"
    source : str
        A label describing the pattern

    """

    source: Optional[str] = None

    span_getter: Optional[SpanGetterArg] = None
    terms: ListOrStr = []
    regex: ListOrStr = []
    regex_attr: Optional[str] = None
    regex_flags: Union[re.RegexFlag, int] = None

    exclude: ExcludeModel = []
    include: IncludeModel = []
    assign: AssignModel = []

    regex_matcher: Optional[Any] = None
    phrase_matcher: Optional[Any] = None


class FullConfig(AsList[SingleConfig]):
    """
    A list of `SingleConfig` objects that should have distinct `source` fields.
    If a single config is passed, it will be automatically converted to a list of
    a single element.
    """

    @classmethod
    def source_uniqueness(cls, v, config=None):
        sources = [item.source for item in v]
        assert len(sources) == len(set(sources)), "Each `source` field should be unique"
        return v

    @classmethod
    def __get_validators__(cls):
        yield cls.validate
        yield cls.source_uniqueness


if TYPE_CHECKING:
    FullConfig = List[SingleConfig]  # noqa: F811
