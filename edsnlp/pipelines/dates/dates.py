from typing import List, Dict, Optional, Union
from dateparser import DateDataParser
import numpy as np
from spacy.language import Language
from spacy.tokens import Token, Span, Doc

from edsnlp.base import BaseComponent
from edsnlp.matchers.regex import RegexMatcher


class Dates(BaseComponent):
    """
    Tags dates.

    Parameters
    ----------
    nlp:
        Language pipeline object
    dates:
        Dictionary containing regular expressions of dates.
    """

    # noinspection PyProtectedMember
    def __init__(
        self,
        nlp: Language,
        dates: List[str],
    ):

        self.nlp = nlp

        self.matcher = RegexMatcher()
        self.matcher.add("date", dates)

        self.parser = DateDataParser(languages=["fr"])

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

        for date in dates:
            date.label_ = self.parser.get_date_data(date.text).date_obj.strftime(
                "%Y-%m-%d"
            )

        doc.spans["dates"] = dates

        return doc
