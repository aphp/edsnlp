from typing import List, Optional, Set, Union

from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipelines.base import SpanGetterArg, get_spans
from edsnlp.pipelines.qualifiers.base import RuleBasedQualifier
from edsnlp.utils.filter import consume_spans, filter_spans
from edsnlp.utils.inclusion import check_inclusion
from edsnlp.utils.resources import get_verbs

from . import patterns


class ReportedSpeechQualifier(RuleBasedQualifier):
    """
    The `eds.reported_speech` component uses a simple rule-based algorithm to detect
    spans that relate to reported speech (eg when the doctor quotes the patient).
    It was designed at AP-HP's EDS.

    Examples
    --------
    The following snippet matches a simple terminology, and checks whether the extracted
    entities are part of a reported speech. It is complete and can be run _as is_.

    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.sentences")
    # Dummy matcher
    nlp.add_pipe(
        "eds.matcher",
        config=dict(terms=dict(patient="patient", alcool="alcoolisé")),
    )
    nlp.add_pipe("eds.reported_speech")

    text = (
        "Le patient est admis aux urgences ce soir pour une douleur au bras. "
        "Il nie être alcoolisé."
    )

    doc = nlp(text)

    doc.ents
    # Out: (patient, alcoolisé)

    doc.ents[0]._.reported_speech
    # Out: False

    doc.ents[1]._.reported_speech
    # Out: True
    ```

    Extensions
    ----------
    The `eds.reported_speech` component declares two extensions, on both `Span` and
    `Token` objects :

    1. The `reported_speech` attribute is a boolean, set to `True` if the component
       predicts that the span/token is reported.
    2. The `reported_speech_` property is a human-readable string, computed from the
       `reported_speech` attribute. It implements a simple getter function that outputs
       `DIRECT` or `REPORTED`, depending on the value of `reported_speech`.

    Parameters
    ----------
    nlp : Language
        spaCy nlp pipeline to use for matching.
    name : Optional[str]
        The component name.
    quotation : str
        String gathering all quotation cues.
    verbs : List[str]
        List of reported speech verbs.
    following : List[str]
        List of terms following a reported speech.
    preceding : List[str]
        List of terms preceding a reported speech.
    attr : str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM",
        or a dict with the key 'term_attr'
        we can also add a key for each regex.
    span_getter : SpanGetterArg
        Which entities should be classified. By default, `doc.ents`
    on_ents_only : Union[bool, str, List[str], Set[str]]
        Whether to look for matches around detected entities only.
        Useful for faster inference in downstream tasks.

        - If True, will look in all ents located in `doc.ents` only
        - If an iterable of string is passed, will additionally look in `doc.spans[key]`
        for each key in the iterable
    within_ents : bool
        Whether to consider cues within entities.
    explain : bool
        Whether to keep track of cues for each entity.

    Authors and citation
    --------------------
    The `eds.reported_speech` component was developed by AP-HP's Data Science team.
    """

    def __init__(
        self,
        nlp: Language,
        name: Optional[str] = "eds.reported_speech",
        *,
        pseudo: Optional[List[str]] = None,
        preceding: Optional[List[str]] = None,
        following: Optional[List[str]] = None,
        quotation: Optional[List[str]] = None,
        verbs: Optional[List[str]] = None,
        attr: str = "NORM",
        span_getter: SpanGetterArg = None,
        on_ents_only: Union[bool, str, List[str], Set[str]] = True,
        within_ents: bool = False,
        explain: bool = False,
    ):

        terms = dict(
            pseudo=pseudo or [],
            preceding=patterns.preceding if preceding is None else preceding,
            following=patterns.following if following is None else following,
            quotation=patterns.quotation if quotation is None else quotation,
            verbs=patterns.verbs if verbs is None else verbs,
        )
        terms["verbs"] = self.load_verbs(terms["verbs"])

        quotation = terms.pop("quotation")

        super().__init__(
            nlp=nlp,
            name=name,
            attr=attr,
            explain=explain,
            terms=terms,
            on_ents_only=on_ents_only,
            span_getter=span_getter,
        )

        self.regex_matcher = RegexMatcher(attr=attr)
        self.regex_matcher.build_patterns(dict(quotation=quotation))

        self.within_ents = within_ents
        self.set_extensions()

    def set_extensions(self) -> None:
        super().set_extensions()

        for cls in (Token, Span):
            if not cls.has_extension("reported_speech"):
                cls.set_extension("reported_speech", default=False)

            if not cls.has_extension("reported_speech_"):
                cls.set_extension(
                    "reported_speech_",
                    getter=lambda token: "REPORTED"
                    if token._.reported_speech
                    else "DIRECT",
                )

        if not Span.has_extension("reported_speech_cues"):
            Span.set_extension("reported_speech_cues", default=[])

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
        matches = self.get_matches(doc)
        matches += list(self.regex_matcher(doc, as_spans=True))

        boundaries = self._boundaries(doc)

        # Removes duplicate matches and pseudo-expressions in one statement
        matches = filter_spans(matches, label_to_remove="pseudo")

        entities = list(get_spans(doc, self.span_getter))
        ents = None

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

            sub_preceding = [m for m in sub_matches if m.label_ == "preceding"]
            sub_following = [m for m in sub_matches if m.label_ == "following"]
            sub_verbs = [m for m in sub_matches if m.label_ == "verbs"]
            sub_quotation = [m for m in sub_matches if m.label_ == "quotation"]

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

                if self.explain and reported_speech:
                    ent._.reported_speech_cues += cues

                if not self.on_ents_only and reported_speech:
                    for token in ent:
                        token._.reported_speech = True

        return doc
