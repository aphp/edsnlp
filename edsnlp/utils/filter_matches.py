from typing import List

from edsnlp.pipelines.generic import GenericMatcher
from spacy.language import Language
from spacy.tokens import Span


def _filter_matches(matches: List[Span], label: str):
    """
    Function to filter matches with a specific label.

    Parameters
    ----------
    matches: List[Span]
        List of matches to filter.
    label: str
        Label used to filter matches.

    Returns
    -------
    List of filtered matches.
    """
    return [match for match in matches if match.label_ == label]
