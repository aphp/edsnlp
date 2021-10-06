from typing import List, Union
from dateparser import DateDataParser
from spacy.language import Language
from spacy.tokens import Span, Doc

from spacy.util import filter_spans

from datetime import datetime, timedelta

from loguru import logger

from edsnlp.base import BaseComponent
from edsnlp.matchers.regex import RegexMatcher


def td2str(td: timedelta):
    """
    Transforms a timedelta object to a string representation.

    Parameters
    ----------
    td : timedelta
        The timedelta object to represent.

    Returns
    -------
    str
        Usable representation for the timedelta object.
    """
    seconds = td.total_seconds()
    days = int(seconds / 3600 / 24)
    return f"TD{days:+d}"


def date_getter(date: Span) -> str:
    """
    Getter for dates. Uses the information from ``note_datetime``.

    Parameters
    ----------
    date : Span
        Date detected by the pipeline.

    Returns
    -------
    str
        Normalized date.
    """

    d = date._.parsed_date

    if d is None:
        # dateparser could not interpret the date.
        return "????-??-??"

    delta = date._.parsed_delta
    note_datetime = date.doc._.note_datetime

    if date.label_ == "absolute":
        normalized = d.strftime("%Y-%m-%d")
    elif date.label_ == "no_year":
        if note_datetime:
            year = note_datetime.strftime("%Y")
        else:
            year = "????"
        normalized = d.strftime(f"{year}-%m-%d")
    else:
        if note_datetime:
            # We need to adjust the timedelta, since most dates are set at 00h00.
            # The slightest difference leads to a day difference.
            d = note_datetime + delta
            normalized = d.strftime("%Y-%m-%d")
        else:
            normalized = td2str(d - datetime.now())

    return normalized


class Dates(BaseComponent):
    """
    Tags dates.

    Parameters
    ----------
    nlp: spacy.language.Language
        Language pipeline object
    absolute: List[str]
        List of regular expressions for absolute dates.
    relative: List[str]
        List of regular expressions for relative dates.
    no_year: List[str]
        List of regular expressions for dates that do not display a year.
    """

    # noinspection PyProtectedMember
    def __init__(
        self,
        nlp: Language,
        absolute: Union[List[str], str],
        relative: Union[List[str], str],
        no_year: Union[List[str], str],
    ):

        logger.warning("``dates`` pipeline is still in beta.")

        self.nlp = nlp

        if isinstance(absolute, str):
            absolute = [absolute]
        if isinstance(relative, str):
            relative = [relative]
        if isinstance(no_year, str):
            no_year = [no_year]

        self.matcher = RegexMatcher(attr="LOWER")
        self.matcher.add("absolute", absolute)
        self.matcher.add("relative", relative)
        self.matcher.add("no_year", no_year)

        self.parser = DateDataParser(languages=["fr"])

        if not Doc.has_extension("note_datetime"):
            Doc.set_extension("note_datetime", default=None)

        if not Span.has_extension("parsed_date"):
            Span.set_extension("parsed_date", default=None)

        if not Span.has_extension("parsed_delta"):
            Span.set_extension("parsed_delta", default=None)

        if not Span.has_extension("date"):
            Span.set_extension("date", getter=date_getter)

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

        dates = self.matcher(doc)

        dates = filter_spans(dates)

        return dates

    # noinspection PyProtectedMember
    def __call__(self, doc: Doc) -> Doc:
        """
        Tags dates.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        doc:
            spaCy Doc object, annotated for dates
        """
        dates = self.process(doc)

        for date in dates:
            d = self.parser.get_date_data(date.text).date_obj

            if d is None:
                date._.parsed_date = None
            else:
                date._.parsed_date = d
                date._.parsed_delta = d - datetime.now() + timedelta(seconds=10)

        doc.spans["dates"] = dates

        return doc
