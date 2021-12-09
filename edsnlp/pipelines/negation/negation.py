from typing import Any, Dict, List, Optional, Union

from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans

from edsnlp.pipelines.matcher import GenericMatcher
from edsnlp.utils.filter import consume_spans
from edsnlp.utils.filter_matches import _filter_matches
from edsnlp.utils.inclusion import check_inclusion


class Negation(GenericMatcher):
    """
    Implements the NegEx algorithm.

    The component looks for four kinds of expressions in the text :

    - preceding negations, ie cues that precede a negated expression
    - following negations, ie cues that follow a negated expression
    - pseudo negations : contain a negation cue, but are not negations (eg "pas de doute"/"no doubt").
    - terminations, ie words that delimit propositions.
      The negation spans from the preceding cue to the termination.

    Inspiration : https://github.com/jenojp/negspacy

    Parameters
    ----------
    nlp: Language
        spaCy nlp pipeline to use for matching.
    pseudo: List[str]
        List of pseudo negation terms.
    preceding: List[str]
        List of preceding negation terms
    following: List[str]
        List of following negation terms.
    termination: List[str]
        List of termination terms.
    verbs: List[str]
        List of negation verbs.
    fuzzy: bool
         Whether to perform fuzzy matching on the terms.
    filter_matches: bool
        Whether to filter out overlapping matches.
    attr: str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with the key 'term_attr'
        we can also add a key for each regex.
    on_ents_only: bool
        Whether to look for matches around detected entities only.
        Useful for faster inference in downstream tasks.
    within_ents: bool
        Whether to consider cues within entities.
    explain: bool
        Whether to keep track of cues for each entity.
    regex: Optional[Dict[str, Union[List[str], str]]]
        A dictionnary of regex patterns.
    fuzzy_kwargs: Optional[Dict[str, Any]]
        Default options for the fuzzy matcher, if used.
    """

    def __init__(
        self,
        nlp: Language,
        pseudo: List[str],
        preceding: List[str],
        following: List[str],
        termination: List[str],
        verbs: List[str],
        fuzzy: bool,
        filter_matches: bool,
        attr: str,
        on_ents_only: bool,
        within_ents: bool,
        regex: Optional[Dict[str, Union[List[str], str]]],
        fuzzy_kwargs: Optional[Dict[str, Any]],
        explain: bool,
        **kwargs,
    ):

        super().__init__(
            nlp,
            terms=dict(
                pseudo=pseudo,
                termination=termination,
                preceding=preceding,
                following=following,
                verbs=self.load_verbs(verbs),
            ),
            fuzzy=fuzzy,
            filter_matches=filter_matches,
            attr=attr,
            on_ents_only=on_ents_only,
            regex=regex,
            fuzzy_kwargs=fuzzy_kwargs,
            **kwargs,
        )

        if not Token.has_extension("negated"):
            Token.set_extension("negated", default=False)

        if not Token.has_extension("polarity_"):
            Token.set_extension(
                "polarity_",
                getter=lambda token: "NEG" if token._.negated else "AFF",
            )

        if not Span.has_extension("negated"):
            Span.set_extension("negated", default=False)

        if not Span.has_extension("negation_cues"):
            Span.set_extension("negation_cues", default=[])

        if not Span.has_extension("polarity_"):
            Span.set_extension(
                "polarity_",
                getter=lambda span: "NEG" if span._.negated else "AFF",
            )

        if not Doc.has_extension("negations"):
            Doc.set_extension("negations", default=[])

        self.explain = explain
        self.within_ents = within_ents

    def load_verbs(self, verbs: List[str]) -> List[str]:
        """
        Conjugate negating verbs to specific tenses.

        Parameters
        ----------
        verbs: list of negating verbs to conjugate

        Returns
        -------
        list_neg_verbs: List of negating verbs conjugated to specific tenses.
        """
        neg_verbs = self._conjugate(verbs)

        neg_verbs = neg_verbs.loc[
            ((neg_verbs["mode"] == "Indicatif") & (neg_verbs["temps"] == "Présent"))
            | (neg_verbs["temps"] == "Participe Présent")
            | (neg_verbs["temps"] == "Participe Passé")
        ]

        list_neg_verbs = list(neg_verbs["variant"].unique())

        return list_neg_verbs

    def __call__(self, doc: Doc) -> Doc:
        """
        Finds entities related to negation.

        Parameters
        ----------
        doc: spaCy Doc object

        Returns
        -------
        doc: spaCy Doc object, annotated for negation
        """

        matches = self.process(doc)

        terminations = _filter_matches(matches, "termination")
        boundaries = self._boundaries(doc, terminations)

        entities = list(doc.ents) + list(doc.spans.get("discarded", []))
        ents = None

        # Removes duplicate matches and pseudo-expressions in one statement
        matches = filter_spans(matches)

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

            sub_preceding = _filter_matches(sub_matches, "preceding")
            sub_following = _filter_matches(sub_matches, "following")
            sub_verbs = _filter_matches(sub_matches, "verbs")

            if not sub_preceding + sub_following + sub_verbs:
                continue

            if not self.on_ents_only:
                for token in doc[start:end]:
                    token._.negated = any(
                        m.end <= token.i for m in sub_preceding + sub_verbs
                    ) or any(m.start > token.i for m in sub_following)

            for ent in ents:

                if self.within_ents:
                    cues = [m for m in sub_preceding + sub_verbs if m.end <= ent.end]
                    cues += [m for m in sub_following if m.start >= ent.start]
                else:
                    cues = [m for m in sub_preceding + sub_verbs if m.end <= ent.start]
                    cues += [m for m in sub_following if m.start >= ent.end]

                negated = ent._.negated or bool(cues)

                ent._.negated = negated

                if self.explain and negated:
                    ent._.negation_cues += cues

                if not self.on_ents_only and negated:
                    for token in ent:
                        token._.negated = True

        return doc
