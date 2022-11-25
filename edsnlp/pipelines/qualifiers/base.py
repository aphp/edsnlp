from itertools import chain
from typing import Dict, List, Optional

from loguru import logger
from spacy.language import Language
from spacy.tokens import Doc, Span

from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.pipelines.base import BaseComponent


def check_normalizer(nlp: Language) -> None:
    components = {name: component for name, component in nlp.pipeline}
    normalizer = components.get("normalizer")

    if normalizer and not normalizer.lowercase:
        logger.warning(
            "You have chosen the NORM attribute, but disabled lowercasing "
            "in your normalisation pipeline. "
            "This WILL hurt performance : you might want to use the "
            "LOWER attribute instead."
        )


def get_qualifier_extensions(nlp: Language):
    """
    Check for all qualifiers present in the pipe and return its corresponding extension
    """
    return {
        name: nlp.get_pipe_meta(name).assigns[0].split("span.")[-1]
        for name, pipe in nlp.pipeline
        if isinstance(pipe, Qualifier)
    }


class Qualifier(BaseComponent):
    """
    Implements the NegEx algorithm.

    Parameters
    ----------
    nlp : Language
        spaCy nlp pipeline to use for matching.
    attr : str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with the key 'term_attr'
        we can also add a key for each regex.
    on_ents_only : bool
        Whether to look for matches around detected entities only.
        Useful for faster inference in downstream tasks.
    explain : bool
        Whether to keep track of cues for each entity.
    **terms : Dict[str, Optional[List[str]]]
        Terms to look for.
    """

    defaults = dict()

    def __init__(
        self,
        nlp: Language,
        attr: str,
        on_ents_only: bool,
        explain: bool,
        **terms: Dict[str, Optional[List[str]]],
    ):

        if attr.upper() == "NORM":
            check_normalizer(nlp)

        self.phrase_matcher = EDSPhraseMatcher(vocab=nlp.vocab, attr=attr)
        self.phrase_matcher.build_patterns(nlp=nlp, terms=terms)

        self.on_ents_only = on_ents_only
        self.explain = explain

    def get_defaults(
        self, **kwargs: Dict[str, Optional[List[str]]]
    ) -> Dict[str, List[str]]:
        """
        Merge terms with their defaults. Null keys are replaced with defaults.

        Returns
        -------
        Dict[str, List[str]]
            Merged dictionary
        """
        # Filter out empty keys
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Update defaults
        terms = self.defaults.copy()
        terms.update(kwargs)

        return terms

    def get_matches(self, doc: Doc) -> List[Span]:
        """
        Extract matches.

        Parameters
        ----------
        doc : Doc
            spaCy `Doc` object.

        Returns
        -------
        List[Span]
            List of detected spans
        """
        if self.on_ents_only:

            sents = set([ent.sent for ent in doc.ents])
            match_iterator = map(
                lambda sent: self.phrase_matcher(sent, as_spans=True), sents
            )

            matches = chain.from_iterable(match_iterator)

        else:
            matches = self.phrase_matcher(doc, as_spans=True)

        return list(matches)

    def __call__(self, doc: Doc) -> Doc:
        return self.process(doc)
