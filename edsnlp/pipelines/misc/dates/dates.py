from itertools import chain
from typing import Dict, List, Optional, Tuple

from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipelines.base import BaseComponent
from edsnlp.utils.filter import filter_spans

from . import patterns
from .parser.models import AbsoluteDate, RelativeDate


class Dates(BaseComponent):
    """
    Tags and normalizes dates, using the open-source `dateparser` library.

    The pipeline uses spaCy's `filter_spans` function.
    It filters out false positives, and introduce a hierarchy between patterns.
    For instance, in case of ambiguity, the pipeline will decide that a date is a
    date without a year rather than a date without a day.

    Parameters
    ----------
    nlp : spacy.language.Language
        Language pipeline object
    absolute : Union[List[str], str]
        List of regular expressions for absolute dates.
    relative : Union[List[str], str]
        List of regular expressions for relative dates
        (eg `hier`, `la semaine prochaine`).
    false_positive : Union[List[str], str]
        List of regular expressions for false positive (eg phone numbers, etc).
    on_ents_only : Union[bool, str, List[str]]
        Wether to look on dates in the whole document or in specific sentences:

        - If `True`: Only look in the sentences of each entity in doc.ents
        - If False: Look in the whole document
        - If given a string `key` or list of string: Only look in the sentences of
          each entity in `#!python doc.spans[key]`
    """

    # noinspection PyProtectedMember
    def __init__(
        self,
        nlp: Language,
        absolute: Optional[List[str]],
        relative: Optional[List[str]],
        false_positive: Optional[List[str]],
        on_ents_only: bool,
        attr: str,
    ):

        self.nlp = nlp

        if absolute is None:
            absolute = patterns.absolute_date_pattern
        if false_positive is None:
            false_positive = patterns.false_positive_pattern
        if relative is None:
            relative = patterns.relative_date_patterns

        if isinstance(absolute, str):
            absolute = [absolute]
        if isinstance(relative, str):
            relative = [relative]
        if isinstance(false_positive, str):
            false_positive = [false_positive]

        self.on_ents_only = on_ents_only
        self.regex_matcher = RegexMatcher(attr=attr, alignment_mode="strict")

        self.regex_matcher.add("false_positive", false_positive)
        self.regex_matcher.add("absolute", absolute)
        self.regex_matcher.add("relative", relative)

        self.set_extensions()

    @staticmethod
    def set_extensions() -> None:

        if not Doc.has_extension("note_datetime"):
            Doc.set_extension("note_datetime", default=None)

        if not Span.has_extension("parsed_date"):
            Span.set_extension("parsed_date", default=None)

        if not Span.has_extension("parsed_delta"):
            Span.set_extension("parsed_delta", default=None)

        if not Span.has_extension("date"):
            Span.set_extension("date", default=None)

    def process(self, doc: Doc) -> List[Span]:
        """
        Find dates in doc.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        dates:
            list of date spans
        """

        if self.on_ents_only:

            if type(self.on_ents_only) == bool:
                ents = doc.ents
            else:
                if type(self.on_ents_only) == str:
                    self.on_ents_only = [self.on_ents_only]
                ents = []
                for key in self.on_ents_only:
                    ents.extend(list(doc.spans[key]))

            dates = []
            for sent in set([ent.sent for ent in ents]):
                dates = chain(
                    dates,
                    self.regex_matcher(
                        sent,
                        as_spans=True,
                        return_groupdict=True,
                    ),
                )

        else:
            dates = self.regex_matcher(
                doc,
                as_spans=True,
                return_groupdict=True,
            )

        dates = filter_spans(dates)
        dates = [date for date in dates if date[0].label_ != "false_positive"]

        return dates

    def parse(self, dates: List[Tuple[Span, Dict[str, str]]]) -> List[Span]:
        """
        Parse dates using the groupdict returned by the matcher.

        Parameters
        ----------
        dates : List[Tuple[Span, Dict[str, str]]]
            List of tuples containing the spans and groupdict
            returned by the matcher.

        Returns
        -------
        List[Span]
            List of processed spans, with the date parsed.
        """

        for span, groupdict in dates:
            if span.label_.startswith("relative"):
                groupdict = RelativeDate.parse_obj(groupdict)
            else:
                groupdict = AbsoluteDate.parse_obj(groupdict)

            span._.date = groupdict

        return [span for span, _ in dates]

    def __call__(self, doc: Doc) -> Doc:
        """
        Tags dates.

        Parameters
        ----------
        doc : Doc
            spaCy Doc object

        Returns
        -------
        doc : Doc
            spaCy Doc object, annotated for dates
        """
        dates = self.process(doc)
        dates = self.parse(dates)

        doc.spans["dates"] = dates

        return doc
