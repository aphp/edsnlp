from typing import Iterable

from spacy.tokens import Doc, Span

from edsnlp.pipelines.ner.scores.base_score import SimpleScoreMatcher


class SofaMatcher(SimpleScoreMatcher):
    def set_extensions(self):
        super().set_extensions()
        if not Span.has_extension("score_method"):
            Span.set_extension("score_method", default=None)

    def process(self, doc: Doc) -> Iterable[Span]:
        """
        Extracts, if available, the value of the score.
        Normalizes the score via the provided `self.score_normalization` method.

        Parameters
        ----------
        doc: Doc
            Document to process

        Returns
        -------
        ents: List[Span]
            List of spaCy's spans, with, if found, an added `score_value` extension
        """

        for ent in super().process(doc):
            assigned = ent._.assigned
            if not assigned:
                continue
            if assigned.get("method_max") is not None:
                method = "Maximum"
            elif assigned.get("method_24h") is not None:
                method = "24H"
            elif assigned.get("method_adm") is not None:
                method = "A l'admission"
            else:
                method = "Non précisée"

            normalized_value = self.score_normalization(assigned["value"])

            if normalized_value is not None:
                ent._.set(self.label, int(normalized_value))
                ent._.score_name = self.label
                ent._.score_method = method

                yield ent
