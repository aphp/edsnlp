from operator import attrgetter
from typing import List, Optional, Tuple

from spacy.tokens import Doc, Span


class BaseComponent(object):
    """
    The `BaseComponent` adds a `set_extensions` method,
    called at the creation of the object.

    It helps decouple the initialisation of the pipeline from
    the creation of extensions, and is particularly usefull when
    distributing EDSNLP on a cluster, since the serialisation mechanism
    imposes that the extensions be reset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.set_extensions()

    @classmethod
    def set_extensions(cls) -> None:
        """
        Set `Doc`, `Span` and `Token` extensions.
        """
        pass

    def get_spans(self, doc: Doc):
        """
        Returns sorted spans of interest depending on the eventual `on_ents_only` value.
        Includes `doc.ents` by default, and adds eventual SpanGroups.
        """
        ents = list(doc.ents) + list(doc.spans.get("discarded", []))

        on_ents_only = getattr(self, "on_ents_only", None)

        if isinstance(on_ents_only, str):
            on_ents_only = [on_ents_only]
        if isinstance(on_ents_only, (set, list)):
            for spankey in set(on_ents_only) & set(doc.spans.keys()):
                ents.extend(doc.spans.get(spankey, []))

        return sorted(list(set(ents)), key=(attrgetter("start", "end")))

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
