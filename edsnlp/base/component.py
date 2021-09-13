from itertools import chain
from typing import List, Optional, Tuple

import mlconjug3
import pandas as pd
from spacy.tokens import Doc, Span

from edsnlp.conjugator import conjugate

from spacy.util import filter_spans

if not Doc.has_extension("note_id"):
    Doc.set_extension("note_id", default=None)


class BaseComponent(object):
    """
    Base component that contains the logic for :

    - boundaries selections
    - match filtering
    - verbs conjugation
    """

    split_on_punctuation = False

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

        if self.split_on_punctuation:
            punctuations = [t.i for t in doc if t.is_punct and "-" not in t.text]
        else:
            punctuations = []

        starts = sent_starts + termination_starts + punctuations + [len(doc)]

        # Remove duplicates
        starts = list(set(starts))

        # Sort starts
        starts.sort()

        boundaries = [(start, end) for start, end in zip(starts[:-1], starts[1:])]

        return boundaries

    @staticmethod
    def _conjugate(verbs: List[str]) -> pd.DataFrame:
        """
        Create a list of conjugated verbs at all tenses from a list of infinitive verbs.

        Parameters
        ----------
        verbs:
            List of infinitive verbs to conjugate

        Returns
        ----------
        pd.DataFrame
            Dataframe of conjugated verbs at all tenses
        """

        df = conjugate(verbs)
        df.columns = [
            "infinitif",
            "mode",
            "temps",
            "personne",
            "variant",
        ]

        return df
