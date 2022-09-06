"""`eds.tnm` pipeline."""
from typing import Dict, List, Optional, Tuple, Union

from pydantic import ValidationError
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipelines.base import BaseComponent
from edsnlp.utils.filter import filter_spans

from . import models, patterns

PERIOD_PROXIMITY_THRESHOLD = 3


class TNM(BaseComponent):
    """
    Tags and normalizes TNM mentions.

    Parameters
    ----------
    nlp : spacy.language.Language
        Language pipeline object
    pattern : Optional[Union[List[str], str]]
        List of regular expressions for TNM mentions.
    attr : str
        spaCy attribute to use
    """

    # noinspection PyProtectedMember
    def __init__(
        self,
        nlp: Language,
        pattern: Optional[Union[List[str], str]],
        attr: str,
    ):

        self.nlp = nlp

        if pattern is None:
            pattern = patterns.tnm_pattern

        if isinstance(pattern, str):
            pattern = [pattern]

        self.regex_matcher = RegexMatcher(attr=attr, alignment_mode="strict")
        self.regex_matcher.add("tnm", pattern)

        self.set_extensions()

    @classmethod
    def set_extensions(cls) -> None:
        """
        Set extensions for the dates pipeline.
        """

        if not Span.has_extension("value"):
            Span.set_extension("value", default=None)

    def process(self, doc: Doc) -> List[Span]:
        """
        Find TNM mentions in doc.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        spans:
            list of tnm spans
        """

        spans = self.regex_matcher(
            doc,
            as_spans=True,
            return_groupdict=True,
        )

        spans = filter_spans(spans)

        return spans

    def parse(self, spans: List[Tuple[Span, Dict[str, str]]]) -> List[Span]:
        """
        Parse dates using the groupdict returned by the matcher.

        Parameters
        ----------
        spans : List[Tuple[Span, Dict[str, str]]]
            List of tuples containing the spans and groupdict
            returned by the matcher.

        Returns
        -------
        List[Span]
            List of processed spans, with the date parsed.
        """

        for span, groupdict in spans:
            try:
                span._.value = models.TNM.parse_obj(groupdict)
            except ValidationError:
                span._.value = models.TNM.parse_obj({})

            span.kb_id_ = span._.value.norm()

        return [span for span, _ in spans]

    def __call__(self, doc: Doc) -> Doc:
        """
        Tags TNM mentions.

        Parameters
        ----------
        doc : Doc
            spaCy Doc object

        Returns
        -------
        doc : Doc
            spaCy Doc object, annotated for TNM
        """
        spans = self.process(doc)
        spans = filter_spans(spans)

        spans = self.parse(spans)

        doc.spans["tnm"] = spans

        ents, discarded = filter_spans(list(doc.ents) + spans, return_discarded=True)

        doc.ents = ents

        if "discarded" not in doc.spans:
            doc.spans["discarded"] = []
        doc.spans["discarded"].extend(discarded)

        return doc
