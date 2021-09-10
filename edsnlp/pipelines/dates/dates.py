from typing import List, Union
from dateparser import DateDataParser
from spacy.language import Language
from spacy.tokens import Span, Doc

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
    """

    # noinspection PyProtectedMember
    def __init__(
        self,
        nlp: Language,
        absolute: Union[List[str], str],
        relative: Union[List[str], str],
    ):

        logger.warning("``dates`` pipeline is still in beta.")

        self.nlp = nlp

        if isinstance(absolute, str):
            absolute = [absolute]
        if isinstance(relative, str):
            relative = [relative]

        self.matcher = RegexMatcher(attr="LOWER")
        self.matcher.add("absolute", absolute)
        self.matcher.add("relative", relative)

        self.parser = DateDataParser(languages=["fr"])

        if not Doc.has_extension("note_datetime"):
            Doc.set_extension("note_datetime", default=None)

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

        dates = self._filter_matches(dates)

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
        note_datetime = doc._.note_datetime

        for date in dates:
            d = self.parser.get_date_data(date.text).date_obj

            if date.label_ == "absolute":
                date.label_ = d.strftime("%Y-%m-%d")
            else:
                if note_datetime:
                    # We need to adjust the timedelta, since most dates are set at 00h00.
                    # The slightest difference leads to a day difference.
                    d = d - datetime.now() + note_datetime + timedelta(seconds=2)
                    date.label_ = d.strftime("%Y-%m-%d")
                else:
                    date.label_ = td2str(d - datetime.now())

        doc.spans["dates"] = dates

        return doc
