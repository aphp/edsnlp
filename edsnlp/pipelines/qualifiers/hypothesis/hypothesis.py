from typing import List, Optional

from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from edsnlp.pipelines.qualifiers.base import Qualifier
from edsnlp.pipelines.terminations import termination
from edsnlp.utils.filter import consume_spans, filter_spans, get_spans
from edsnlp.utils.inclusion import check_inclusion
from edsnlp.utils.resources import get_verbs

from .patterns import following, preceding, pseudo, verbs_eds, verbs_hyp


class Hypothesis(Qualifier):
    """
    Hypothesis detection with spaCy.

    The component looks for five kinds of expressions in the text :

    - preceding hypothesis, ie cues that precede a hypothetic expression
    - following hypothesis, ie cues that follow a hypothetic expression
    - pseudo hypothesis : contain a hypothesis cue, but are not hypothesis
      (eg "pas de doute"/"no doubt")
    - hypothetic verbs : verbs indicating hypothesis (eg "douter")
    - classic verbs conjugated to the conditional, thus indicating hypothesis

    Parameters
    ----------
    nlp : Language
        spaCy nlp pipeline to use for matching.
    pseudo : Optional[List[str]]
        List of pseudo hypothesis cues.
    preceding : Optional[List[str]]
        List of preceding hypothesis cues
    following : Optional[List[str]]
        List of following hypothesis cues.
    verbs_hyp : Optional[List[str]]
        List of hypothetic verbs.
    verbs_eds : Optional[List[str]]
        List of mainstream verbs.
    filter_matches : bool
        Whether to filter out overlapping matches.
    attr : str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with the key 'term_attr'
        we can also add a key for each regex.
    on_ents_only : bool
        Whether to look for matches around detected entities only.
        Useful for faster inference in downstream tasks.
    within_ents : bool
        Whether to consider cues within entities.
    explain : bool
        Whether to keep track of cues for each entity.
    regex : Optional[Dict[str, Union[List[str], str]]]
        A dictionnary of regex patterns.
    """

    defaults = dict(
        following=following,
        preceding=preceding,
        pseudo=pseudo,
        termination=termination,
        verbs_eds=verbs_eds,
        verbs_hyp=verbs_hyp,
    )

    def __init__(
        self,
        nlp: Language,
        attr: str,
        pseudo: Optional[List[str]],
        preceding: Optional[List[str]],
        following: Optional[List[str]],
        termination: Optional[List[str]],
        verbs_eds: Optional[List[str]],
        verbs_hyp: Optional[List[str]],
        on_ents_only: bool,
        within_ents: bool,
        explain: bool,
    ):

        terms = self.get_defaults(
            pseudo=pseudo,
            preceding=preceding,
            following=following,
            termination=termination,
            verbs_eds=verbs_eds,
            verbs_hyp=verbs_hyp,
        )
        terms["verbs_preceding"], terms["verbs_following"] = self.load_verbs(
            verbs_hyp=terms.pop("verbs_hyp"),
            verbs_eds=terms.pop("verbs_eds"),
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
    def set_extensions(cls) -> None:
        if not Token.has_extension("hypothesis"):
            Token.set_extension("hypothesis", default=False)

        if not Token.has_extension("hypothesis_"):
            Token.set_extension(
                "hypothesis_",
                getter=lambda token: "HYP" if token._.hypothesis else "CERT",
            )

        if not Span.has_extension("hypothesis"):
            Span.set_extension("hypothesis", default=False)

        if not Span.has_extension("hypothesis_"):
            Span.set_extension(
                "hypothesis_",
                getter=lambda span: "HYP" if span._.hypothesis else "CERT",
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
        """
        Finds entities related to hypothesis.

        Parameters
        ----------
        doc: spaCy Doc object

        Returns
        -------
        doc: spaCy Doc object, annotated for hypothesis
        """

        matches = self.get_matches(doc)

        terminations = get_spans(matches, "termination")
        boundaries = self._boundaries(doc, terminations)

        # Removes duplicate matches and pseudo-expressions in one statement
        matches = filter_spans(matches, label_to_remove="pseudo")

        entities = list(doc.ents) + list(doc.spans.get("discarded", []))
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

            sub_preceding = get_spans(sub_matches, "preceding")
            sub_following = get_spans(sub_matches, "following")
            sub_preceding += get_spans(sub_matches, "verbs_preceding")
            sub_following += get_spans(sub_matches, "verbs_following")

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
