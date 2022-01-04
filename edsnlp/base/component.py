from typing import List, Optional, Tuple

from spacy.tokens import Doc, Span
from spacy.util import filter_spans


class BaseComponent(object):
    """
    Base component that contains the logic for :

    - boundaries selections
    - match filtering
    - verbs conjugation
    """

    @staticmethod
    def _filter_matches(matches: List[Span]) -> List[Span]:
        """
        Filter matches to remove duplicates and inclusions.

        Arguments
        ---------
        matches: List of matches (spans).

        Returns
        -------
        filtered_matches: List of filtered matches.
        """

        return filter_spans(matches)

    def _boundaries(
        self, doc: Doc, terminations: Optional[List[Span]] = None
    ) -> List[Tuple[int, int]]:
        """
        Create sub sentences based sentences and terminations found in text.

        Parameters
        ----------
        doc:
            spaCy Doc object
        terminations:
            List of tuples with (match_id, start, end)

        Returns
        -------
        boundaries:
            List of tuples with (start, end) of spans
        """

        if terminations is None:
            terminations = []

        sent_starts = [sent.start for sent in doc.sents]
        termination_starts = [t.start for t in terminations]

        starts = sent_starts + termination_starts + [len(doc)]

        # Remove duplicates
        starts = list(set(starts))

        # Sort starts
        starts.sort()

        boundaries = [(start, end) for start, end in zip(starts[:-1], starts[1:])]

        return boundaries
