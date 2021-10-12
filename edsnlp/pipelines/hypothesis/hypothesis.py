from typing import Any, Dict, List, Optional, Union

from spacy.language import Language
from spacy.tokens import Doc, Span, Token

from edsnlp.pipelines.matcher import GenericMatcher
from edsnlp.utils.filter_matches import _filter_matches
from edsnlp.utils.inclusion import check_inclusion


class Hypothesis(GenericMatcher):
    """
    Hypothesis detection with Spacy.

    The component looks for five kinds of expressions in the text :

    - preceding hypothesis, ie cues that precede a hypothetic expression
    - following hypothesis, ie cues that follow a hypothetic expression
    - pseudo hypothesis : contain a hypothesis cue, but are not hypothesis (eg "pas de doute"/"no doubt")
    - hypothetic verbs : verbs indicating hypothesis (eg "douter")
    - classic verbs conjugated to the conditional, thus indicating hypothesis

    Parameters
    ----------
    nlp: Language
        spaCy nlp pipeline to use for matching.
    pseudo: List[str]
        List of pseudo hypothesis cues.
    preceding: List[str]
        List of preceding hypothesis cues
    following: List[str]
        List of following hypothesis cues.
    verbs_hyp: List[str]
        List of hypothetic verbs.
    verbs_eds: List[str]
        List of mainstream verbs.
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
        confirmation: List[str],
        preceding: List[str],
        following: List[str],
        verbs_hyp: List[str],
        verbs_eds: List[str],
        fuzzy: bool,
        filter_matches: bool,
        attr: str,
        explain: bool,
        on_ents_only: bool,
        regex: Optional[Dict[str, Union[List[str], str]]],
        fuzzy_kwargs: Optional[Dict[str, Any]],
        **kwargs,
    ):

        super().__init__(
            nlp,
            terms=dict(
                pseudo=pseudo,
                confirmation=confirmation,
                preceding=preceding,
                following=following,
                verbs=self.load_verbs(verbs_hyp, verbs_eds),
            ),
            fuzzy=fuzzy,
            filter_matches=filter_matches,
            attr=attr,
            on_ents_only=on_ents_only,
            regex=regex,
            fuzzy_kwargs=fuzzy_kwargs,
            **kwargs,
        )

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

        self.explain = explain

    def load_verbs(self, verbs_hyp: List[str], verbs_eds: List[str]) -> List[str]:
        """
        Conjugate "classic" verbs to conditional, and add hypothesis verbs conjugated to all tenses.

        Parameters
        ----------
        verbs_hyp: List of verbs that specifically imply an hypothesis.
        verbs_eds: List of general verbs.

        Returns
        -------
        list of hypothesis verbs conjugated at all tenses and classic verbs conjugated to conditional.
        """
        classic_verbs = self._conjugate(verbs_eds)
        classic_verbs = classic_verbs.loc[classic_verbs["mode"] == "Conditionnel"]
        list_classic_verbs = list(classic_verbs["variant"].unique())

        hypo_verbs = self._conjugate(verbs_hyp)
        list_hypo_verbs = list(hypo_verbs["variant"].unique())

        return list_hypo_verbs + list_classic_verbs

    def __call__(self, doc: Doc) -> Doc:
        """
        Finds entities related to hypothesis.

        Parameters
        ----------
        doc: spaCy Doc object

        Returns
        -------
        doc: spaCy Doc object, annotated for hypothesis
        """

        matches = self.process(doc)
        boundaries = self._boundaries(doc)

        # confirmation = _filter_matches(matches, "confirmation")
        pseudo = _filter_matches(matches, "pseudo")
        preceding = _filter_matches(matches, "preceding")
        following = _filter_matches(matches, "following")
        verbs = _filter_matches(matches, "verbs")

        true_matches = []

        for match in matches:
            if match.label_ in {"pseudo", "confirmation"}:
                continue
            pseudo_flag = False
            for p in pseudo:
                if match.start >= p.start and match.end <= p.end:
                    pseudo_flag = True
                    break
            if not pseudo_flag:
                true_matches.append(match)

        for start, end in boundaries:
            ents = [ent for ent in doc.ents if check_inclusion(ent, start, end)]

            if self.on_ents_only and not ents:
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
                    token._.hypothesis = any(
                        m.end <= token.i for m in sub_preceding + sub_verbs
                    ) or any(m.start > token.i for m in sub_following)
            for ent in ents:

                cues = [m for m in sub_preceding + sub_verbs if m.end <= ent.start]
                cues += [m for m in sub_following if m.start > ent.end]

                hypothesis = ent._.hypothesis or bool(cues)

                ent._.hypothesis = hypothesis

                if self.explain and hypothesis:
                    ent._.hypothesis_cues += cues

                if not self.on_ents_only and hypothesis:
                    for token in ent:
                        token._.hypothesis = True

        return doc
