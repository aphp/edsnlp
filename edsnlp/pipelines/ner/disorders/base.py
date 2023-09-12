import re
from typing import Any, Dict, List, Union

from spacy import Language
from spacy.tokens import Doc, Span

from edsnlp.pipelines.base import SpanSetterArg
from edsnlp.pipelines.core.contextual_matcher import ContextualMatcher
from edsnlp.utils.deprecation import deprecated_getter_factory
from edsnlp.utils.filter import filter_spans


class DisorderMatcher(ContextualMatcher):
    """
    Base class used to implement various disorders or behaviors extraction pipes

    Parameters
    ----------
    nlp : Language
        spaCy `Language` object.
    name : str
        The name of the pipe
    patterns: Union[Dict[str, Any], List[Dict[str, Any]]]
        The configuration dictionary
    include_assigned : bool
        Whether to include (eventual) assign matches to the final entity
    ignore_excluded : bool
        Whether to skip excluded tokens during matching.
    ignore_space_tokens: bool
        Whether to skip space tokens during matching.
    detailed_status_mapping: Dict[int, str]
        Mapping from integer status (0, 1 or 2) to human-readable string

    alignment_mode : str
        Overwrite alignment mode.
    regex_flags : Union[re.RegexFlag, int]
        RegExp flags to use when matching, filtering and assigning (See
        the [re docs](https://docs.python.org/3/library/re.html#flags))
    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        *,
        label: str,
        patterns: Union[Dict[str, Any], List[Dict[str, Any]]],
        include_assigned: bool = True,
        ignore_excluded: bool = True,
        ignore_space_tokens: bool = True,
        detailed_status_mapping: Dict[int, str] = {
            0: "ABSENT",
            1: "PRESENT",
        },
        alignment_mode: str = "expand",
        regex_flags: Union[re.RegexFlag, int] = re.S,
        span_setter: SpanSetterArg,
    ):
        self.nlp = nlp
        self.detailed_status_mapping = detailed_status_mapping

        super().__init__(
            nlp=nlp,
            name=name,
            label=label,
            attr="NORM",
            patterns=patterns,
            ignore_excluded=ignore_excluded,
            ignore_space_tokens=ignore_space_tokens,
            regex_flags=regex_flags,
            alignment_mode=alignment_mode,
            assign_as_span=True,
            include_assigned=include_assigned,
            span_setter=span_setter,
        )

    def set_extensions(self) -> None:
        super().set_extensions()

        if not Span.has_extension("status"):
            Span.set_extension("status", default=1)
        if not Span.has_extension("detailed_status"):
            Span.set_extension("detailed_status", default="PRESENT")
        if not Span.has_extension("detailled_status"):
            Span.set_extension(
                "detailled_status",
                getter=deprecated_getter_factory("detailed_status", "detailed_status"),
            )

    def __call__(self, doc: Doc) -> Doc:
        """
        Tags entities.

        Parameters
        ----------
        doc : Doc
            spaCy Doc object

        Returns
        -------
        doc : Doc
            annotated spaCy Doc object
        """
        spans = list(self.process(doc))
        for span in spans:
            span._.detailed_status = self.detailed_status_mapping[span._.status]

        self.set_spans(doc, filter_spans(spans))

        return doc
