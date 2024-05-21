import re
from typing import TYPE_CHECKING, Any, List, Optional, Union

from pydantic import BaseModel, Extra, root_validator

from edsnlp.matchers.utils import ListOrStr
from edsnlp.utils.span_getters import Context, SentenceContext, SpanGetterArg
from edsnlp.utils.typing import AsList

Flags = Union[re.RegexFlag, int]


def validate_window(cls, values):
    if isinstance(values.get("regex"), str):
        values["regex"] = [values["regex"]]
    window = values.get("window")
    if window is None or isinstance(window, (int, tuple, list)):
        values["limit_to_sentence"] = True
    window = values.get("window")
    if window is not None:
        values["window"] = Context.validate(window)
    if values.get("limit_to_sentence"):
        values["window"] = values.get("window") & SentenceContext(0, 0)
    return values


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
    """
    A dictionary  to define exclusion rules. Exclusion rules are given as Regexes, and
    if a match is found in the surrounding context of an extraction, the extraction is
    removed. Each dictionary should have the following keys:

    Parameters
    ----------
    regex: ListOrStr
        A single Regex or a list of Regexes
    window: Optional[Context]
        Size of the context to use (in number of words). You can provide the window as:

            - A [context string][context-string]
            - A positive integer, in this case the used context will be taken **after**
              the extraction
            - A negative integer, in this case the used context will be taken **before**
              the extraction
            - A tuple of integers `(start, end)`, in this case the used context will be
              the snippet from `start` tokens before the extraction to `end` tokens
              after the extraction
    limit_to_sentence: Optional[bool]
        If set to `True`, the exclusion will be limited to the sentence containing the
        extraction
    regex_flags: Optional[Flags]
        Flags to use when compiling the Regexes
    regex_attr: Optional[str]
        An attribute to overwrite the given `attr` when matching with Regexes.
    """

    regex: ListOrStr = []
    limit_to_sentence: Optional[bool] = None
    window: Optional[Context] = None
    regex_flags: Optional[Flags] = None
    regex_attr: Optional[str] = None
    matcher: Optional[Any] = None

    validate_window = root_validator(pre=True, allow_reuse=True)(validate_window)


class SingleIncludeModel(BaseModel):
    """
    A dictionary  to define inclusion rules. Inclusion rules are given as Regexes, and
    if a match isn't found in the surrounding context of an extraction, the extraction
    is removed. Each dictionary should have the following keys:

    Parameters
    ----------
    regex: ListOrStr
        A single Regex or a list of Regexes
    window: Optional[Context]
        Size of the context to use (in number of words). You can provide the window as:

            - A [context string][context-string]
            - A positive integer, in this case the used context will be taken **after**
              the extraction
            - A negative integer, in this case the used context will be taken **before**
              the extraction
            - A tuple of integers `(start, end)`, in this case the used context will be
              the snippet from `start` tokens before the extraction to `end` tokens
              after the extraction
    limit_to_sentence: Optional[bool]
        If set to `True`, the exclusion will be limited to the sentence containing the
        extraction
    regex_flags: Optional[Flags]
        Flags to use when compiling the Regexes
    regex_attr: Optional[str]
        An attribute to overwrite the given `attr` when matching with Regexes.
    """

    regex: ListOrStr = []
    limit_to_sentence: Optional[bool] = None
    window: Optional[Context] = None
    regex_flags: Optional[Flags] = None
    regex_attr: Optional[str] = None
    matcher: Optional[Any] = None

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
    name: ListOrStr
        A name (string)
    window: Optional[Context]
        Size of the context to use (in number of words). You can provide the window as:

        - A [context string][context-string]
        - A positive integer, in this case the used context will be taken **after**
          the extraction
        - A negative integer, in this case the used context will be taken **before**
          the extraction
        - A tuple of integers `(start, end)`, in this case the used context will be the
          snippet from `start` tokens before the extraction to `end` tokens after the
          extraction
    span_getter: Optional[SpanGetterArg]
        A span getter to pick the assigned spans from already extracted entities
        in the doc.
    regex: Optional[Context]
        A dictionary where keys are labels and values are **Regexes with a single
        capturing group**
    replace_entity: Optional[bool]
        If set to `True`, the match from the corresponding assign key will be used as
        entity, instead of the main match.
        See [this paragraph][the-replace_entity-parameter]
    reduce_mode: Optional[Flags]
        Set how multiple assign matches are handled. See the documentation of the
        [`reduce_mode` parameter][the-reduce_mode-parameter]
    required: Optional[str]
        If set to `True`, the assign key must match for the extraction to be kept. If
        it does not match, the extraction is discarded.
    """

    name: str
    regex: ListOrStr = []
    span_getter: Optional[SpanGetterArg] = None
    limit_to_sentence: Optional[bool] = None
    window: Optional[Context] = None
    regex_flags: Optional[Flags] = None
    regex_attr: Optional[str] = None
    replace_entity: bool = False
    reduce_mode: Optional[str] = None
    required: Optional[bool] = False

    matcher: Optional[Any] = None

    validate_window = root_validator(pre=True, allow_reuse=True)(validate_window)


class AssignModel(AsList[SingleAssignModel]):
    """
    A list of `SingleAssignModel` objects that should have at most
    one element with `replace_entity=True`. If a single config is passed,
    it will be automatically converted to a list of a single element.
    """

    @classmethod
    def name_uniqueness(cls, v, config):
        names = [item.name for item in v]
        assert len(names) == len(set(names)), "Each `name` field should be unique"
        return v

    @classmethod
    def replace_uniqueness(cls, v, config):
        replace = [item for item in v if item.replace_entity]
        assert (
            len(replace) <= 1
        ), "Only 1 assign element can be set with `replace_entity=True`"
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


class SingleConfig(BaseModel, extra=Extra.forbid):
    """
    A single configuration for the contextual matcher.

    Parameters
    ----------
    source : str
        A label describing the pattern
    regex : ListOrStr
        A single Regex or a list of Regexes
    regex_attr : Optional[str]
        An attributes to overwrite the given `attr` when matching with Regexes.
    terms : Union[re.RegexFlag, int]
        A single term or a list of terms (for exact matches)
    exclude : AsList[SingleExcludeModel]
        ??? subdoc "One or more exclusion patterns"

            ::: edsnlp.pipes.core.contextual_matcher.models.SingleExcludeModel
                options:
                    only_parameters: "no-header"
    assign : AsList[SingleAssignModel]
        ??? subdoc "One or more assignment patterns"

            ::: edsnlp.pipes.core.contextual_matcher.models.SingleAssignModel
                options:
                    only_parameters: "no-header"

    """

    source: str
    terms: ListOrStr = []
    regex: ListOrStr = []
    regex_attr: Optional[str] = None
    regex_flags: Union[re.RegexFlag, int] = None
    exclude: ExcludeModel = []
    include: IncludeModel = []
    assign: AssignModel = []


class FullConfig(AsList[SingleConfig]):
    """
    A list of `SingleConfig` objects that should have distinct `source` fields.
    If a single config is passed, it will be automatically converted to a list of
    a single element.
    """

    @classmethod
    def source_uniqueness(cls, v, config):
        sources = [item.source for item in v]
        assert len(sources) == len(set(sources)), "Each `source` field should be unique"
        return v

    @classmethod
    def __get_validators__(cls):
        yield cls.validate
        yield cls.source_uniqueness


if TYPE_CHECKING:
    FullConfig = List[SingleConfig]  # noqa: F811
