from typing import Any, Dict, Generator, List, Optional, Union

from importlib_metadata import re
from spacy import Language
from spacy.tokens import Doc, Span

from edsnlp.pipelines.core.contextual_matcher import ContextualMatcher
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
    status_mapping: Optional[Dict[int, str]]
        Mapping from integer status (0, 1 or 2) to human-readable string

    alignment_mode : str
        Overwrite alignment mode.
    regex_flags : Union[re.RegexFlag, int]
        RegExp flags to use when matching, filtering and assigning (See
        [here](https://docs.python.org/3/library/re.html#flags))

    """

    def __init__(
        self,
        nlp: Language,
        name: str,
        patterns: Union[Dict[str, Any], List[Dict[str, Any]]],
        include_assigned: bool = True,
        ignore_excluded: bool = True,
        ignore_space_tokens: bool = True,
        status_mapping: Optional[Dict[int, str]] = None,
    ):
        self.nlp = nlp
        self.status_mapping = status_mapping or {0: "ABSENT", 1: "PRESENT"}

        super().__init__(
            nlp=nlp,
            name=name,
            attr="NORM",
            patterns=patterns,
            ignore_excluded=ignore_excluded,
            ignore_space_tokens=ignore_space_tokens,
            regex_flags=re.S,
            alignment_mode="expand",
            assign_as_span=True,
            include_assigned=include_assigned,
        )

        self.set_extensions()

    @classmethod
    def set_extensions(cl) -> None:
        super().set_extensions()

        if not Span.has_extension("status"):
            Span.set_extension("status", default=1)
        if not Span.has_extension("status_"):
            Span.set_extension("status_", default="PRESENT")

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
        spans = self.postprocess(doc, self.process(doc))
        spans = filter_spans(spans)

        for span in spans:
            span._.status_ = self.status_mapping[span._.status]

        doc.spans[self.name] = spans

        return doc

    def postprocess(self, doc: Doc, spans: Generator[Span, None, None]):
        """
        Can be overrid
        """
        yield from spans
