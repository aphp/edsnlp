from typing import List, Optional, Set, Union

from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from edsnlp.pipelines.base import SpanGetterArg, get_spans
from edsnlp.pipelines.qualifiers.base import RuleBasedQualifier
from edsnlp.pipelines.terminations import termination as default_termination
from edsnlp.utils.deprecation import deprecated_getter_factory
from edsnlp.utils.filter import consume_spans, filter_spans
from edsnlp.utils.inclusion import check_inclusion
from edsnlp.utils.resources import get_verbs

from . import patterns


class NegationQualifier(RuleBasedQualifier):
    """
    The `eds.negation` component uses a simple rule-based algorithm to detect negated
    spans. It was designed at AP-HP's EDS, following the insights of the NegEx algorithm
    by [@chapman_simple_2001].

    The component looks for five kinds of expressions in the text :

    - preceding negations, i.e., cues that precede a negated expression
    - following negations, i.e., cues that follow a negated expression
    - pseudo negations : contain a negation cue, but are not negations
      (eg "pas de doute"/"no doubt")
    - negation verbs, i.e., verbs that indicate a negation
    - terminations, i.e., words that delimit propositions.
      The negation spans from the preceding cue to the termination.

    Examples
    --------
    The following snippet matches a simple terminology, and checks the polarity of the
    extracted entities. It is complete and can be run _as is_.

    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.sentences")
    # Dummy matcher
    nlp.add_pipe(
        "eds.matcher",
        config=dict(terms=dict(patient="patient", fracture="fracture")),
    )
    nlp.add_pipe("eds.negation")

    text = (
        "Le patient est admis le 23 août 2021 pour une douleur au bras. "
        "Le scanner ne détecte aucune fracture."
    )

    doc = nlp(text)

    doc.ents
    # Out: (patient, fracture)

    doc.ents[0]._.negation  # (1)
    # Out: False

    doc.ents[1]._.negation
    # Out: True
    ```

    1. The result of the component is kept in the `negation` custom extension.

    Extensions
    ----------
    The `eds.negation` component declares two extensions, on both `Span` and `Token`
    objects :

    1. The `negation` attribute is a boolean, set to `True` if the component predicts
       that the span/token is negated.
    2. The `negation_` property is a human-readable string, computed from the `negation`
       attribute. It implements a simple getter function that outputs `AFF` or `NEG`,
       depending on the value of `negation`.

    Performance
    -----------
    The component's performance is measured on three datasets :

    - The ESSAI ([@dalloux2017ESSAI]) and CAS ([@grabar2018CAS]) datasets were developed
      at the CNRS. The two are concatenated.
    - The NegParHyp corpus was specifically developed at AP-HP to test the component
      on actual clinical notes, using pseudonymised notes from the AP-HP.

    | Dataset   | Negation F1 |
    |-----------|-------------|
    | CAS/ESSAI | 71%         |
    | NegParHyp | 88%         |

    !!! note "NegParHyp corpus"

        The NegParHyp corpus was built by matching a subset of the MeSH terminology with
        around 300 documents from AP-HP's clinical data warehouse. Matched entities were
        then labelled for negation, speculation and family context.

    Parameters
    ----------
    nlp : Language
        The pipeline object.
    name : Optional[str]
        The component name.
    attr : str
        spaCy's attribute to use
    pseudo : Optional[List[str]]
        List of pseudo negation cues.
    preceding : Optional[List[str]]
        List of preceding negation cues
    following : Optional[List[str]]
        List of following negation cues.
    verbs : Optional[List[str]]
        List of negation verbs.
    termination : Optional[List[str]]
        List of termination terms.
    span_getter : SpanGetterArg
        Which entities should be classified. By default, `doc.ents`
    on_ents_only : Union[bool, str, List[str], Set[str]]
        Deprecated, use `span_getter` instead.

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
    The `eds.negation` component was developed by AP-HP's Data Science team.
    """

    def __init__(
        self,
        nlp: Language,
        name: Optional[str] = "eds.negation",
        *,
        pseudo: Optional[List[str]] = None,
        preceding: Optional[List[str]] = None,
        following: Optional[List[str]] = None,
        verbs: Optional[List[str]] = None,
        termination: Optional[List[str]] = None,
        attr: str = "NORM",
        span_getter: SpanGetterArg = None,
        on_ents_only: Union[bool, str, List[str], Set[str]] = True,
        within_ents: bool = False,
        explain: bool = False,
    ):
        terms = dict(
            pseudo=patterns.pseudo if pseudo is None else pseudo,
            preceding=patterns.preceding if preceding is None else preceding,
            following=patterns.following if following is None else following,
            termination=default_termination if termination is None else termination,
            verbs=patterns.verbs if verbs is None else verbs,
        )
        terms["verbs_preceding"], terms["verbs_following"] = self.load_verbs(
            terms["verbs"]
        )

        super().__init__(
            nlp=nlp,
            name=name,
            attr=attr,
            explain=explain,
            terms=terms,
            on_ents_only=on_ents_only,
            span_getter=span_getter,
        )

        self.within_ents = within_ents
        self.set_extensions()

    def set_extensions(self) -> None:
        super().set_extensions()
        for cls in (Token, Span):
            if not cls.has_extension("negation"):
                cls.set_extension("negation", default=False)

            if not cls.has_extension("negated"):
                cls.set_extension(
                    "negated", getter=deprecated_getter_factory("negated", "negation")
                )

            if not cls.has_extension("negation_"):
                cls.set_extension(
                    "negation_",
                    getter=lambda token: "NEG" if token._.negation else "AFF",
                )

            if not cls.has_extension("polarity_"):
                cls.set_extension(
                    "polarity_",
                    getter=deprecated_getter_factory("polarity_", "negation_"),
                )

        if not Span.has_extension("negation_cues"):
            Span.set_extension("negation_cues", default=[])

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

        return list_neg_verbs_preceding, list_neg_verbs_following

    def process(self, doc: Doc) -> Doc:
        matches = self.get_matches(doc)

        terminations = [m for m in matches if m.label_ == "termination"]
        boundaries = self._boundaries(doc, terminations)

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
            # Verbs preceding negated content
            sub_preceding += [m for m in sub_matches if m.label_ == "verbs_preceding"]
            # Verbs following negated content
            sub_following += [m for m in sub_matches if m.label_ == "verbs_following"]

            if not sub_preceding + sub_following:
                continue

            if not self.on_ents_only:
                for token in doc[start:end]:
                    token._.negation = any(
                        m.end <= token.i for m in sub_preceding
                    ) or any(m.start > token.i for m in sub_following)

            for ent in ents:
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

        return doc
