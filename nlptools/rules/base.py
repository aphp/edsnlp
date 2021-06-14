from itertools import chain
from typing import List, Optional, Tuple

from spacy.tokens import Doc, Span

if not Doc.has_extension('note_id'):
    Doc.set_extension('note_id', default=None)


class BaseComponent(object):
    """
    Base component that contains the logic for :
    - boundaries selections
    - match filtering
    """

    split_on_punctuation = True

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

        filtered_matches = []

        for span in matches:

            if not set(range(span.start, span.end)).intersection(
                    chain(*[set(range(s.start, s.end)) for s in filtered_matches])
            ):
                filtered_matches.append(span)

            else:
                s = set(range(span.start, span.end))

                for match in filtered_matches[:]:
                    m = set(range(match.start, match.end))

                    if m & s:
                        tokens = sorted(list(s | m))
                        start, end = tokens[0], tokens[-1]

                        new_span = Span(span.doc, start, end + 1, label=span.label_)

                        filtered_matches.remove(match)
                        filtered_matches.append(new_span)
                        break

        return filtered_matches

    def _boundaries(self, doc: Doc, terminations: Optional[List[Span]] = None) -> List[Tuple[int, int]]:
        """
        Create sub sentences based sentences and terminations found in text.
        
        Parameters
        ----------
        doc: spaCy Doc object
        terminations: List of tuples with (match_id, start, end)
            
        Returns
        -------
        boundaries: List of tuples with (start, end) of spans
        """

        if terminations is None:
            terminations = []

        sent_starts = [sent.start for sent in doc.sents]
        termination_starts = [t.start for t in terminations]

        if self.split_on_punctuation:
            punctuations = [t.i for t in doc if t.is_punct and '-' not in t.text]
        else:
            punctuations = []

        starts = sent_starts + termination_starts + punctuations + [len(doc)]

        # Remove duplicates
        starts = list(set(starts))

        # Sort starts
        starts.sort()

        boundaries = [(start, end) for start, end in zip(starts[:-1], starts[1:])]

        return boundaries
