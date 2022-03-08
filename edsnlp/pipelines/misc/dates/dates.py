import re
from datetime import datetime, timedelta
from itertools import chain
from typing import Dict, Generator, Iterable, List, Optional, Tuple

from dateparser import DateDataParser
from dateparser_data.settings import default_parsers
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipelines.base import BaseComponent
from edsnlp.utils.filter import filter_spans

from . import patterns
from .parsing import day2int, month2int, str2int


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
    Getter for dates. Uses the information from `note_datetime`.

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

    if date.label_ in {"absolute", "full_date", "no_day"}:
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
        "RETURN_AS_TIMEZONE_AWARE": False,
    },
)

parser2 = DateDataParser(
    languages=["fr"],
    settings={
        "PREFER_DAY_OF_MONTH": "first",
        "PREFER_DATES_FROM": "past",
        "PARSERS": ["relative-time"],
        "RETURN_AS_TIMEZONE_AWARE": False,
    },
)


def date_parser(text_date: str) -> datetime:
    """
    Function to parse dates. It try first all available parsers
    ('timestamp', 'custom-formats', 'absolute-time') but 'relative-time'.
    If no date is found, retries with 'relative-time'.

    When just the year is identified, it returns a datetime object with
    month and day equal to 1.


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


def apply_groupdict(
    dates: Iterable[Tuple[Span, Dict[str, str]]]
) -> Generator[Span, None, None]:
    for span, groupdict in dates:
        span._.groupdict = groupdict
        yield span


def parse_groupdict(
    day: str = None,
    month: str = None,
    year: str = None,
    hour: str = None,
    minute: str = None,
    second: str = None,
    **kwargs: Dict[str, str],
) -> Dict[str, int]:
    """
    Parse date groupdict.

    Parameters
    ----------
    day : str, optional
        String representation of the day, by default None
    month : str, optional
        String representation of the month, by default None
    year : str, optional
        String representation of the year, by default None
    hour : str, optional
        String representation of the hour, by default None
    minute : str, optional
        String representation of the minute, by default None
    second : str, optional
        String representation of the minute, by default None

    Returns
    -------
    Dict[str, int]
        Parsed groupdict.
    """

    result = dict()

    if day is not None:
        result["day"] = day2int(day)

    if month is not None:
        result["month"] = month2int(month)

    if year is not None:
        result["year"] = str2int(year)

    if hour is not None:
        result["hour"] = str2int(hour)

    if minute is not None:
        result["minute"] = str2int(minute)

    if second is not None:
        result["second"] = str2int(second)

    result.update(**kwargs)

    return result


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
    full : Union[List[str], str]
        List of regular expressions for full dates in YYYY-MM-DD format.
    relative : Union[List[str], str]
        List of regular expressions for relative dates
        (eg `hier`, `la semaine prochaine`).
    no_year : Union[List[str], str]
        List of regular expressions for dates that do not display a year.
    no_day : Union[List[str], str]
        List of regular expressions for dates that do not display a day.
    year_only : Union[List[str], str]
        List of regular expressions for dates that only display a year.
    current : Union[List[str], str]
        List of regular expressions for dates that relate to
        the current month, week, year, etc.
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
        full: Optional[List[str]],
        relative: Optional[List[str]],
        no_year: Optional[List[str]],
        no_day: Optional[List[str]],
        year_only: Optional[List[str]],
        current: Optional[List[str]],
        false_positive: Optional[List[str]],
        on_ents_only: bool,
        attr: str,
    ):

        self.nlp = nlp

        if no_year is None:
            no_year = patterns.no_year_pattern
        if year_only is None:
            year_only = patterns.full_year_pattern
        if no_day is None:
            no_day = patterns.no_day_pattern
        if absolute is None:
            absolute = patterns.absolute_date_pattern
        if relative is None:
            relative = patterns.relative_date_pattern
        if full is None:
            full = patterns.full_date_pattern
        if current is None:
            current = patterns.current_pattern
        if false_positive is None:
            false_positive = patterns.false_positive_pattern

        if isinstance(absolute, str):
            absolute = [absolute]
        if isinstance(relative, str):
            relative = [relative]
        if isinstance(no_year, str):
            no_year = [no_year]
        if isinstance(no_day, str):
            no_day = [no_day]
        if isinstance(year_only, str):
            year_only = [year_only]
        if isinstance(full, str):
            full = [full]
        if isinstance(current, str):
            current = [current]
        if isinstance(false_positive, str):
            false_positive = [false_positive]

        self.on_ents_only = on_ents_only
        self.regex_matcher = RegexMatcher(attr=attr, alignment_mode="strict")

        self.regex_matcher.add("false_positive", false_positive)
        self.regex_matcher.add("full_date", full)
        self.regex_matcher.add("absolute", absolute)
        self.regex_matcher.add("relative", relative)
        self.regex_matcher.add("no_year", no_year)
        self.regex_matcher.add("no_day", no_day)
        self.regex_matcher.add("year_only", year_only)
        self.regex_matcher.add("current", current)

        self.parser = date_parser
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
                        # return_groupdict=True,
                    ),
                )

        else:
            dates = self.regex_matcher(
                doc,
                as_spans=True,
                # return_groupdict=True,
            )

        # dates = apply_groupdict(dates)

        dates = filter_spans(dates)
        dates = [date for date in dates if date.label_ != "false_positive"]

        return dates

    def get_date(self, date: Span) -> Optional[datetime]:
        """
        Get normalised date using `dateparser`.

        Parameters
        ----------
        date : Span
            Date span.

        Returns
        -------
        Optional[datetime]
            If a date is recognised, returns a Python `datetime` object.
            Returns `None` otherwise.
        """

        text_date = date.text

        if date.label_ == "no_day":
            text_date = "01/" + re.sub(r"[\.\/\s]", "/", text_date)

        elif date.label_ == "full_date":
            text_date = re.sub(r"[\.\/\s]", "-", text_date)

            try:
                return datetime.strptime(text_date, "%Y-%m-%d")
            except ValueError:
                try:
                    return datetime.strptime(text_date, "%Y-%d-%m")
                except ValueError:
                    return None

        # text_date = re.sub(r"\.", "-", text_date)

        return self.parser(text_date)

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
