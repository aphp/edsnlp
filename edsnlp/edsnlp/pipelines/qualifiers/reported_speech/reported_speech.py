from typing import List, Optional

from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipelines.qualifiers.base import Qualifier
from edsnlp.utils.filter import consume_spans, filter_spans, get_spans
from edsnlp.utils.inclusion import check_inclusion
from edsnlp.utils.resources import get_verbs

from .patterns import following, preceding, quotation, verbs


class ReportedSpeech(Qualifier):
    """
    Implements a reported speech detection algorithm.

    The components looks for terms indicating patient statements,
    and quotations to detect patient speech.

    Parameters
    ----------
    nlp : Language
        spaCy nlp pipeline to use for matching.
    quotation : str
        String gathering all quotation cues.
    verbs : List[str]
        List of reported speech verbs.
    following : List[str]
        List of terms following a reported speech.
    preceding : List[str]
        List of terms preceding a reported speech.
    filter_matches : bool
        Whether to filter out overlapping matches.
    attr : str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM",
        or a dict with the key 'term_attr'
        we can also add a key for each regex.
    on_ents_only : bool
        Whether to look for matches around detected entities only.
        Useful for faster inference in downstream tasks.
    within_ents : bool
        Whether to consider cues within entities.
    explain : bool
        Whether to keep track of cues for each entity.
    """

    defaults = dict(
        following=following,
        preceding=preceding,
        verbs=verbs,
        quotation=quotation,
    )

    def __init__(
        self,
        nlp: Language,
        attr: str,
        pseudo: Optional[List[str]],
        preceding: Optional[List[str]],
        following: Optional[List[str]],
        quotation: Optional[List[str]],
        verbs: Optional[List[str]],
        on_ents_only: bool,
        within_ents: bool,
        explain: bool,
    ):

        terms = self.get_defaults(
            pseudo=pseudo,
            preceding=preceding,
            following=following,
            quotation=quotation,
            verbs=verbs,
        )
        terms["verbs"] = self.load_verbs(terms["verbs"])

        quotation = terms.pop("quotation")

        super().__init__(
            nlp=nlp,
            attr=attr,
            on_ents_only=on_ents_only,
            explain=explain,
            **terms,
        )

        self.regex_matcher = RegexMatcher(attr=attr)
        self.regex_matcher.build_patterns(dict(quotation=quotation))

        self.within_ents = within_ents

        self.set_extensions()

    @classmethod
    def set_extensions(cls) -> None:

        if not Token.has_extension("reported_speech"):
            Token.set_extension("reported_speech", default=False)

        if not Token.has_extension("reported_speech_"):
            Token.set_extension(
                "reported_speech_",
                getter=lambda token: "REPORTED"
                if token._.reported_speech
                else "DIRECT",
            )

        if not Span.has_extension("reported_speech"):
            Span.set_extension("reported_speech", default=False)

        if not Span.has_extension("reported_speech_"):
            Span.set_extension(
                "reported_speech_",
                getter=lambda span: "REPORTED" if span._.reported_speech else "DIRECT",
            )

        if not Span.has_extension("reported_speech_cues"):
            Span.set_extension("reported_speech_cues", default=[])

        if not Doc.has_extension("rspeechs"):
            Doc.set_extension("rspeechs", default=[])

    def load_verbs(self, verbs: List[str]) -> List[str]:
        """
        Conjugate reporting verbs to specific tenses (trhid person)

        Parameters
        ----------
        verbs: list of reporting verbs to conjugate

        Returns
        -------
        list_rep_verbs: List of reporting verbs conjugated to specific tenses.
        """

        rep_verbs = get_verbs(verbs)

        rep_verbs = rep_verbs.loc[
            (
                (rep_verbs["mode"] == "Indicatif")
                & (rep_verbs["tense"] == "Présent")
                & (rep_verbs["person"].isin(["3s", "3p"]))
            )
            | (rep_verbs["tense"] == "Participe Présent")
            | (rep_verbs["tense"] == "Participe Passé")
        ]

        list_rep_verbs = list(rep_verbs["term"].unique())

        return list_rep_verbs

    def process(self, doc: Doc) -> Doc:
        """
        Finds entities related to reported speech.

        Parameters
        ----------
        doc: spaCy Doc object

        Returns
        -------
        doc: spaCy Doc object, annotated for negation
        """

        matches = self.get_matches(doc)
        matches += list(self.regex_matcher(doc, as_spans=True))

        boundaries = self._boundaries(doc)

        entities = list(doc.ents) + list(doc.spans.get("discarded", []))
        ents = None

        # Removes duplicate matches and pseudo-expressions in one statement
        matches = filter_spans(matches, label_to_remove="pseudo")

        for start, end in boundaries:

            ents, entities = consume_spans(
                entities,
                filter=lambda s: check_inclusion(s, start, end),
                second_chance=ents,
            )

            sub_matches, matches = consume_spans(
                matches, lambda s: start <= s.start < end
            )

            if self.on_ents_only and not ents:
                continue

            sub_preceding = get_spans(sub_matches, "preceding")
            sub_following = get_spans(sub_matches, "following")
            sub_verbs = get_spans(sub_matches, "verbs")
            sub_quotation = get_spans(sub_matches, "quotation")

            if not sub_preceding + sub_following + sub_verbs + sub_quotation:
                continue

            if not self.on_ents_only:
                for token in doc[start:end]:
                    token._.reported_speech = (
                        any(m.end <= token.i for m in sub_preceding + sub_verbs)
                        or any(m.start > token.i for m in sub_following)
                        or any(
                            ((m.start < token.i) & (m.end > token.i + 1))
                            for m in sub_quotation
                        )
                    )
            for ent in ents:

                if self.within_ents:
                    cues = [m for m in sub_preceding + sub_verbs if m.end <= ent.end]
                    cues += [m for m in sub_following if m.start >= ent.start]
                else:
                    cues = [m for m in sub_preceding + sub_verbs if m.end <= ent.start]
                    cues += [m for m in sub_following if m.start >= ent.end]

                cues += [
                    m
                    for m in sub_quotation
                    if (m.start < ent.start) & (m.end > ent.end)
                ]

                reported_speech = ent._.reported_speech or bool(cues)
                ent._.reported_speech = reported_speech

                if self.explain:
                    ent._.reported_speech_cues += cues

                if not self.on_ents_only and reported_speech:
                    for token in ent:
                        token._.reported_speech = True
        return doc
