from typing import List, Union, Dict, Any, Optional

from edsnlp.pipelines.generic import GenericMatcher
from spacy.language import Language
from spacy.tokens import Token, Span, Doc

from edsnlp.utils.filter_matches import _filter_matches


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
        regex: Optional[Dict[str, Union[List[str], str]]],
        fuzzy_kwargs: Optional[Dict[str, Any]],
        **kwargs,
    ):

        super().__init__(
            nlp,
            terms=dict(
                pseudo=pseudo,
                preceding=preceding,
                following=following,
                termination=termination,
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

        if not Span.has_extension("polarity_"):
            Span.set_extension(
                "polarity_",
                getter=lambda span: "NEG" if span._.negated else "AFF",
            )

        if not Doc.has_extension("negations"):
            Doc.set_extension("negations", default=[])

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
        pseudo = _filter_matches(matches, "pseudo")
        preceding = _filter_matches(matches, "preceding")
        following = _filter_matches(matches, "following")
        verbs = _filter_matches(matches, "verbs")

        boundaries = self._boundaries(doc, terminations)

        true_matches = []

        for match in matches:
            if match.label_ in {"pseudo", "termination"}:
                continue
            pseudo_flag = False
            for p in pseudo:
                if match.start >= p.start and match.end <= p.end:
                    pseudo_flag = True
                    break
            if not pseudo_flag:
                true_matches.append(match)

        for start, end in boundaries:
            if self.on_ents_only and not doc[start:end].ents:
                continue

            sub_preceding = [
                m
                for m in preceding
                if ((start <= m.start < end) and (m in true_matches))
            ]

            sub_following = [
                m
                for m in following
                if ((start <= m.start < end) and (m in true_matches))
            ]

            sub_verbs = [m for m in verbs if (start <= m.start < end)]

            if not sub_preceding + sub_following + sub_verbs:
                continue

            if not self.on_ents_only:
                for token in doc[start:end]:
                    token._.negated = any(
                        m.end <= token.i for m in sub_preceding + sub_verbs
                    ) or any(m.start > token.i for m in sub_following)
            for ent in doc[start:end].ents:
                ent._.negated = (
                    ent._.negated
                    or any(m.end <= ent.start for m in sub_preceding + sub_verbs)
                    or any(m.start > ent.end for m in sub_following)
                )

        return doc
