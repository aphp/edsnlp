import re
from datetime import datetime, timedelta
from typing import List, Optional, Union

from dateparser import DateDataParser
from dateparser_data.settings import default_parsers
from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

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

    if date.label_ in {"absolute", "full_date"}:
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


parsers = [parser for parser in default_parsers if parser != "relative-time"]
parser1 = DateDataParser(
    languages=["fr"],
    settings={
        "PREFER_DAY_OF_MONTH": "first",
        "PREFER_DATES_FROM": "past",
        "PARSERS": parsers,
    },
)

parser2 = DateDataParser(
    languages=["fr"],
    settings={
        "PREFER_DAY_OF_MONTH": "first",
        "PREFER_DATES_FROM": "past",
        "PARSERS": ["relative-time"],
    },
)


def date_parser(text_date: str) -> datetime:
    """Function to parse dates. It try first all available parsers ('timestamp', 'custom-formats', 'absolute-time') but 'relative-time'.
    If no date is found, retries with 'relative-time'.

    When just the year is identified. It returns a datetime object with month and day equal to 1.


    Parameters
    ----------
    text_date : str

    Returns
    -------
    datetime
    """

    parsed_date = parser1.get_date_data(text_date)
    if parsed_date.date_obj:
        if parsed_date.period == "year":
            return datetime(year=parsed_date.date_obj.year, month=1, day=1)
        else:
            return parsed_date.date_obj
    else:
        parsed_date2 = parser2.get_date_data(text_date)
        return parsed_date2.date_obj


class Dates(BaseComponent):
    """
    Tags dates.

    Parameters
    ----------
    nlp: spacy.language.Language
        Language pipeline object
    absolute: List[str]
        List of regular expressions for absolute dates.
    full_date: List[str]
        List of regular expressions for full dates in YYYY-MM-DD format.
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
        full_date: Union[List[str], str],
        relative: Union[List[str], str],
        no_year: Union[List[str], str],
        false_positive: Union[List[str], str],
    ):

        logger.warning("``dates`` pipeline is still in beta.")

        self.nlp = nlp

        if isinstance(absolute, str):
            absolute = [absolute]
        if isinstance(relative, str):
            relative = [relative]
        if isinstance(no_year, str):
            no_year = [no_year]
        if isinstance(full_date, str):
            full_date = [full_date]
        if isinstance(false_positive, str):
            false_positive = [false_positive]

        self.matcher = RegexMatcher(attr="LOWER", alignment_mode="expand")
        self.matcher.add("absolute", absolute)
        self.matcher.add("full_date", full_date)
        self.matcher.add("relative", relative)
        self.matcher.add("no_year", no_year)
        self.matcher.add("false_positive", false_positive)

        self.parser = date_parser

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
        dates = [date for date in dates if date.label_ != "false_positive"]

        return dates

    def get_date(self, date: Span) -> Optional[datetime]:
        """
        Get normalised date using ``dateparser``.

        Parameters
        ----------
        date : Span
            Date span.

        Returns
        -------
        Optional[datetime]
            If a date is recognised, returns a Python ``datetime`` object. Returns ``None`` otherwise.
        """

        text_date = date.text

        if date.label_ == "full_date":
            text_date = re.sub(r"[\.\/\s]", "-", text_date)

            return datetime.strptime(text_date, "%Y-%m-%d")
        else:
            text_date = re.sub(r"\.", "-", text_date)

            return self.parser(text_date)

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
            d = self.get_date(date)

            if d is None:
                date._.parsed_date = None
            else:
                date._.parsed_date = d
                date._.parsed_delta = d - datetime.now() + timedelta(seconds=10)

        doc.spans["dates"] = dates

        return doc
