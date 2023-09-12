import re
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from spacy import registry
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.pipelines.base import SpanSetterArg
from edsnlp.pipelines.core.contextual_matcher import ContextualMatcher


class SimpleScoreMatcher(ContextualMatcher):
    """
    Matcher component to extract a numeric score

    Parameters
    ----------
    nlp : Language
        The pipeline object
    label : str
        The name of the extracted score
    regex : List[str]
        A list of regexes to identify the score
    attr : str
        Whether to match on the text ('TEXT') or on the normalized text ('NORM')
    value_extract : str
        Regex with capturing group to get the score value
    score_normalization : Union[str, Callable[[Union[str,None]], Any]]
        Function that takes the "raw" value extracted from the `value_extract`
        regex and should return:

        - None if no score could be extracted
        - The desired score value else
    window : int
        Number of token to include after the score's mention to find the
        score's value
    ignore_excluded : bool
        Whether to ignore excluded spans when matching
    ignore_space_tokens : bool
        Whether to ignore space tokens when matching
    flags : Union[re.RegexFlag, int]
        Regex flags to use when matching
    score_name: str
        Deprecated, use `label` instead. The name of the extracted score
    label : str
        Label name to use for the `Span` object and the extension
    span_setter: Optional[SpanSetterArg]
        How to set matches on the doc
    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        *,
        regex: List[str] = None,
        attr: str = "NORM",
        value_extract: Union[str, Dict[str, str], List[Dict[str, str]]] = None,
        score_normalization: Union[str, Callable[[Union[str, None]], Any]] = None,
        window: int = 7,
        ignore_excluded: bool = False,
        ignore_space_tokens: bool = False,
        flags: Union[re.RegexFlag, int] = 0,
        score_name: str = None,
        label: str = None,
        span_setter: Optional[SpanSetterArg] = None,
    ):
        if score_name is not None:
            warnings.warn(
                "`score_name` is deprecated, use `label` instead.",
                DeprecationWarning,
            )
            label = score_name

        if label is None:
            raise ValueError("`label` parameter is required.")

        if span_setter is None:
            span_setter = {"ents": True, label: True}

        if isinstance(value_extract, str):
            value_extract = dict(
                name="value",
                regex=value_extract,
                window=window,
            )

        if isinstance(value_extract, dict):
            value_extract = [value_extract]

        value_exists = False
        for i, extract in enumerate(value_extract):
            extract["window"] = extract.get("window", window)
            if extract.get("name", None) == "value":
                value_exists = True
                extract["replace_entity"] = True
                extract["reduce_mode"] = "keep_first"
            value_extract[i] = extract

        assert value_exists, "You should provide a `value` regex in the `assign` dict."

        patterns = dict(
            source=label,
            regex=regex,
            assign=value_extract,
        )

        super().__init__(
            nlp=nlp,
            name=name,
            patterns=patterns,
            assign_as_span=False,
            alignment_mode="expand",
            ignore_excluded=ignore_excluded,
            ignore_space_tokens=ignore_space_tokens,
            attr=attr,
            regex_flags=flags,
            include_assigned=False,
            label=label,
            span_setter=span_setter,
        )

        if isinstance(score_normalization, str):
            self.score_normalization = registry.get("misc", score_normalization)
        else:
            self.score_normalization = score_normalization

    def set_extensions(self) -> None:
        super().set_extensions()
        if not Span.has_extension(self.label):
            Span.set_extension(self.label, default=None)
        if not Span.has_extension("score_name"):
            Span.set_extension("score_name", default=None)
        if not Span.has_extension("score_value"):
            Span.set_extension("score_value", getter=lambda x: x._.value)

    def process(self, doc: Doc) -> Iterable[Span]:
        """
        Extracts, if available, the value of the score.
        Normalizes the score via the provided `self.score_normalization` method.

        Parameters
        ----------
        doc: Doc
            Document to process

        Yields
        ------
        Span
            Matches with, if found, an added `score_value` extension
        """
        for ent in super().process(doc):
            value = ent._.assigned.get("value", None)
            if value is None:
                continue
            normalized_value = self.score_normalization(value)
            if normalized_value is not None:
                ent._.score_name = self.label
                ent._.set(self.label, normalized_value)

                yield ent
