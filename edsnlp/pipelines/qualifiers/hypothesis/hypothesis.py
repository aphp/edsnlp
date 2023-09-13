from typing import List, Optional, Set, Union

from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from edsnlp.pipelines.base import SpanGetterArg, get_spans
from edsnlp.pipelines.qualifiers.base import RuleBasedQualifier
from edsnlp.pipelines.terminations import termination as default_termination
from edsnlp.utils.filter import consume_spans, filter_spans
from edsnlp.utils.inclusion import check_inclusion
from edsnlp.utils.resources import get_verbs

from . import patterns


class HypothesisQualifier(RuleBasedQualifier):
    """
    The `eds.hypothesis` pipeline uses a simple rule-based algorithm to detect spans
    that are speculations rather than certain statements.

    The component looks for five kinds of expressions in the text :

    - preceding hypothesis, ie cues that precede a hypothetical expression
    - following hypothesis, ie cues that follow a hypothetical expression
    - pseudo hypothesis : contain a hypothesis cue, but are not hypothesis
      (eg "pas de doute"/"no doubt")
    - hypothetical verbs : verbs indicating hypothesis (eg "douter")
    - classic verbs conjugated to the conditional, thus indicating hypothesis

    Examples
    --------
    The following snippet matches a simple terminology, and checks whether the extracted
    entities are part of a speculation. It is complete and can be run _as is_.

    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.sentences")
    # Dummy matcher
    nlp.add_pipe(
        "eds.matcher",
        config=dict(terms=dict(douleur="douleur", fracture="fracture")),
    )
    nlp.add_pipe("eds.hypothesis")

    text = (
        "Le patient est admis le 23 août 2021 pour une douleur au bras. "
        "Possible fracture du radius."
    )

    doc = nlp(text)

    doc.ents
    # Out: (douleur, fracture)

    doc.ents[0]._.hypothesis
    # Out: False

    doc.ents[1]._.hypothesis
    # Out: True
    ```

    Extensions
    ----------
    The `eds.hypothesis` component declares two extensions, on both `Span` and `Token`
    objects :

    1. The `hypothesis` attribute is a boolean, set to `True` if the component predicts
       that the span/token is a speculation.
    2. The `hypothesis_` property is a human-readable string, computed from the
       `hypothesis` attribute. It implements a simple getter function that outputs
       `HYP` or `CERT`, depending on the value of `hypothesis`.

    Performance
    ------------
    The component's performance is measured on three datasets :

    - The ESSAI ([@dalloux2017ESSAI]) and CAS ([@grabar2018CAS]) datasets were developed
      at the CNRS. The two are concatenated.
    - The NegParHyp corpus was specifically developed at APHP's CDW to test the
      component on actual clinical notes, using pseudonymised notes from the APHP's CDW.

    | Dataset   | Hypothesis F1 |
    | --------- | ------------- |
    | CAS/ESSAI | 49%           |
    | NegParHyp | 52%           |

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
        List of pseudo hypothesis cues.
    preceding : Optional[List[str]]
        List of preceding hypothesis cues
    following : Optional[List[str]]
        List of following hypothesis cues.
    verbs_hyp : Optional[List[str]]
        List of hypothetical verbs.
    verbs_eds : Optional[List[str]]
        List of mainstream verbs.
    termination : Optional[List[str]]
        List of termination terms.
    attr : str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with the key 'term_attr'
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
    The `eds.hypothesis` pipeline was developed by AP-HP's Data Science team.
    """

    def __init__(
        self,
        nlp: Language,
        name: Optional[str] = "eds.hypothesis",
        *,
        pseudo: Optional[List[str]] = None,
        preceding: Optional[List[str]] = None,
        following: Optional[List[str]] = None,
        verbs_eds: Optional[List[str]] = None,
        verbs_hyp: Optional[List[str]] = None,
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
            verbs_eds=patterns.verbs_eds if verbs_eds is None else verbs_eds,
            verbs_hyp=patterns.verbs_hyp if verbs_hyp is None else verbs_hyp,
        )
        terms["verbs_preceding"], terms["verbs_following"] = self.load_verbs(
            verbs_hyp=terms.pop("verbs_hyp"),
            verbs_eds=terms.pop("verbs_eds"),
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
            if not cls.has_extension("hypothesis"):
                cls.set_extension("hypothesis", default=False)

            if not cls.has_extension("hypothesis_"):
                cls.set_extension(
                    "hypothesis_",
                    getter=lambda token: "HYP" if token._.hypothesis else "CERT",
                )

        if not Span.has_extension("hypothesis_cues"):
            Span.set_extension("hypothesis_cues", default=[])

        if not Doc.has_extension("hypothesis"):
            Doc.set_extension("hypothesis", default=[])

    def load_verbs(
        self,
        verbs_hyp: List[str],
        verbs_eds: List[str],
    ) -> List[str]:
        """
        Conjugate "classic" verbs to conditional, and add hypothesis
        verbs conjugated to all tenses.

        Parameters
        ----------
        verbs_hyp: List of verbs that specifically imply an hypothesis.
        verbs_eds: List of general verbs.

        Returns
        -------
        list of hypothesis verbs conjugated at all tenses and classic
        verbs conjugated to conditional.
        """

        classic_verbs = get_verbs(verbs_eds)
        classic_verbs = classic_verbs.loc[classic_verbs["mode"] == "Conditionnel"]
        list_classic_verbs = list(classic_verbs["term"].unique())

        hypo_verbs = get_verbs(verbs_hyp)
        list_hypo_verbs_preceding = list(hypo_verbs["term"].unique())

        hypo_verbs_following = hypo_verbs.loc[hypo_verbs["tense"] == "Participe Passé"]
        list_hypo_verbs_following = list(hypo_verbs_following["term"].unique())

        return (
            list_hypo_verbs_preceding + list_classic_verbs,
            list_hypo_verbs_following,
        )

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
                    token._.hypothesis = any(
                        m.end <= token.i for m in sub_preceding
                    ) or any(m.start > token.i for m in sub_following)

            for ent in ents:
                if self.within_ents:
                    cues = [m for m in sub_preceding if m.end <= ent.end]
                    cues += [m for m in sub_following if m.start >= ent.start]
                else:
                    cues = [m for m in sub_preceding if m.end <= ent.start]
                    cues += [m for m in sub_following if m.start >= ent.end]

                hypothesis = ent._.hypothesis or bool(cues)

                ent._.hypothesis = hypothesis

                if self.explain and hypothesis:
                    ent._.hypothesis_cues += cues

                if not self.on_ents_only and hypothesis:
                    for token in ent:
                        token._.hypothesis = True

        return doc
