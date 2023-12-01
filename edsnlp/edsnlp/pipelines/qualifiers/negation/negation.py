from typing import List, Optional

from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from edsnlp.pipelines.qualifiers.base import Qualifier
from edsnlp.pipelines.terminations import termination
from edsnlp.utils.deprecation import deprecated_getter_factory
from edsnlp.utils.filter import consume_spans, filter_spans, get_spans
from edsnlp.utils.inclusion import check_inclusion
from edsnlp.utils.resources import get_verbs

from .patterns import following, preceding, pseudo, verbs


class Negation(Qualifier):
    """
    Implements the NegEx algorithm.

    The component looks for five kinds of expressions in the text :

    - preceding negations, ie cues that precede a negated expression

    - following negations, ie cues that follow a negated expression

    - pseudo negations : contain a negation cue, but are not negations
      (eg "pas de doute"/"no doubt")

    - negation verbs, ie verbs that indicate a negation

    - terminations, ie words that delimit propositions.
      The negation spans from the preceding cue to the termination.

    Parameters
    ----------
    nlp : Language
        spaCy nlp pipeline to use for matching.
    attr : str
        spaCy's attribute to use
    pseudo : Optional[List[str]]
        List of pseudo negation terms.
    preceding : Optional[List[str]]
        List of preceding negation terms
    following : Optional[List[str]]
        List of following negation terms.
    termination : Optional[List[str]]
        List of termination terms.
    verbs : Optional[List[str]]
        List of negation verbs.
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
        pseudo=pseudo,
        verbs=verbs,
        termination=termination,
    )

    def __init__(
        self,
        nlp: Language,
        attr: str,
        pseudo: Optional[List[str]],
        preceding: Optional[List[str]],
        following: Optional[List[str]],
        termination: Optional[List[str]],
        verbs: Optional[List[str]],
        on_ents_only: bool,
        within_ents: bool,
        explain: bool,
    ):

        terms = self.get_defaults(
            pseudo=pseudo,
            preceding=preceding,
            following=following,
            termination=termination,
            verbs=verbs,
        )
        terms["verbs_preceding"], terms["verbs_following"] = self.load_verbs(
            terms["verbs"]
        )

        super().__init__(
            nlp=nlp,
            attr=attr,
            on_ents_only=on_ents_only,
            explain=explain,
            **terms,
        )

        self.within_ents = within_ents
        self.set_extensions()

    @classmethod
    def set_extensions(cl) -> None:

        if not Token.has_extension("negation"):
            Token.set_extension("negation", default=False)

        if not Token.has_extension("negated"):
            Token.set_extension(
                "negated", getter=deprecated_getter_factory("negated", "negation")
            )

        if not Token.has_extension("negation_"):
            Token.set_extension(
                "negation_",
                getter=lambda token: "NEG" if token._.negation else "AFF",
            )

        if not Token.has_extension("polarity_"):
            Token.set_extension(
                "polarity_",
                getter=deprecated_getter_factory("polarity_", "negation_"),
            )

        if not Span.has_extension("negation"):
            Span.set_extension("negation", default=False)

        if not Span.has_extension("negated"):
            Span.set_extension(
                "negated", getter=deprecated_getter_factory("negated", "negation")
            )

        if not Span.has_extension("negation_cues"):
            Span.set_extension("negation_cues", default=[])

        if not Span.has_extension("negation_"):
            Span.set_extension(
                "negation_",
                getter=lambda span: "NEG" if span._.negation else "AFF",
            )

        if not Span.has_extension("polarity_"):
            Span.set_extension(
                "polarity_",
                getter=deprecated_getter_factory("polarity_", "negation_"),
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
        list_neg_verbs_preceding: List of conjugated negating verbs preceding entities.
        list_neg_verbs_following: List of conjugated negating verbs following entities.
        """

        neg_verbs = get_verbs(verbs)

        neg_verbs_preceding = neg_verbs.loc[
            ((neg_verbs["mode"] == "Indicatif") & (neg_verbs["tense"] == "Présent"))
            | (neg_verbs["tense"] == "Participe Présent")
            | (neg_verbs["tense"] == "Participe Passé")
            | (neg_verbs["tense"] == "Infinitif Présent")
        ]
        neg_verbs_following = neg_verbs.loc[neg_verbs["tense"] == "Participe Passé"]
        list_neg_verbs_preceding = list(neg_verbs_preceding["term"].unique())
        list_neg_verbs_following = list(neg_verbs_following["term"].unique())

        return (list_neg_verbs_preceding, list_neg_verbs_following)

    def annotate_entity(
        self,
        ent: Span,
        sub_preceding: List[Span],
        sub_following: List[Span],
    ) -> None:
        """
        Annotate entities using preceding and following negations.

        Parameters
        ----------
        ent : Span
            Entity to annotate
        sub_preceding : List[Span]
            List of preceding negations cues
        sub_following : List[Span]
            List of following negations cues
        """
        if self.within_ents:
            cues = [m for m in sub_preceding if m.end <= ent.end]
            cues += [m for m in sub_following if m.start >= ent.start]
        else:
            cues = [m for m in sub_preceding if m.end <= ent.start]
            cues += [m for m in sub_following if m.start >= ent.end]

        negation = ent._.negation or bool(cues)

        ent._.negation = negation

        if self.explain and negation:
            ent._.negation_cues += cues

        if not self.on_ents_only and negation:
            for token in ent:
                token._.negation = True

    def process(self, doc: Doc) -> Doc:
        """
        Finds entities related to negation.

        Parameters
        ----------
        doc: spaCy `Doc` object

        Returns
        -------
        doc: spaCy `Doc` object, annotated for negation
        """

        matches = self.get_matches(doc)

        terminations = get_spans(matches, "termination")
        boundaries = self._boundaries(doc, terminations)

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
            # Verbs preceding negated content
            sub_preceding += get_spans(sub_matches, "verbs_preceding")
            # Verbs following negated content
            sub_following += get_spans(sub_matches, "verbs_following")

            if not sub_preceding + sub_following:
                continue

            if not self.on_ents_only:
                for token in doc[start:end]:
                    token._.negation = any(
                        m.end <= token.i for m in sub_preceding
                    ) or any(m.start > token.i for m in sub_following)

            for ent in ents:
                self.annotate_entity(
                    ent=ent,
                    sub_preceding=sub_preceding,
                    sub_following=sub_following,
                )

        return doc

    def __call__(self, doc: Doc) -> Doc:
        return self.process(doc)
