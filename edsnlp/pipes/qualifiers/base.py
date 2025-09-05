import warnings
from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Optional, Set, Union

from spacy.tokens import Doc, Span, Token

from edsnlp.core import PipelineProtocol
from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.pipes.base import (
    BaseSpanAttributeClassifierComponent,
    SpanGetterArg,
    validate_span_getter,
)


def check_normalizer(nlp: PipelineProtocol) -> None:
    components = {name: component for name, component in nlp.pipeline}
    normalizer = components.get("normalizer")

    if normalizer and not normalizer.lowercase:
        warnings.warn(
            "You have chosen the NORM attribute, but disabled lowercasing "
            "in your normalisation pipeline. "
            "This WILL hurt performance : you might want to use the "
            "LOWER attribute instead."
        )


@dataclass
class BaseTokenQualifierResults:
    """
    Base dataclass to store qualification informations for each token.
    Specific qualifier pipes should inherit from this class and
    add specific fields as needed
    """

    token: Token


@dataclass
class BaseEntQualifierResults:
    """
    Base dataclass to store qualification informations for each entity.
    Specific qualifier pipes should inherit from this class and
    add specific fields as needed
    """

    ent: Span


@dataclass
class BaseQualifierResults:
    """
    Base dataclass to store qualification informations for all tokens and entities.
    """

    tokens: List[BaseTokenQualifierResults]
    ents: List[BaseEntQualifierResults]


class RuleBasedQualifier(BaseSpanAttributeClassifierComponent):
    """
    Implements the ConText algorithm (eq. NegEx for negations) for detecting contextual
    attributes text.

    Parameters
    ----------
    nlp : PipelineProtocol
        The pipeline object.
    attr : str
        spaCy's attribute to use:
        a string with the value "TEXT" or "NORM", or a dict with the key 'term_attr'
        we can also add a key for each regex.
    span_getter : SpanGetterArg
        Which entities should be classified. By default, `doc.ents`
    on_ents_only : Union[bool, str, List[str], Set[str]]
        Whether to look for matches around detected entities only.
        Useful for faster inference in downstream tasks.

        - If True, will look in all ents located in `doc.ents` only
        - If an iterable of string is passed, will additionally look in `doc.spans[key]`
        for each key in the iterable
    explain : bool
        Whether to keep track of cues for each entity.
    **terms : Dict[str, Optional[List[str]]]
        Terms to look for.
    """

    defaults = dict()

    def __init__(
        self,
        nlp: PipelineProtocol,
        name: Optional[str] = None,
        *,
        attr: str,
        attributes: List[str],
        span_getter: SpanGetterArg,
        on_ents_only: Union[bool, str, List[str], Set[str]],
        explain: bool,
        terms: Dict[str, Optional[List[str]]],
        regex: Dict[str, Optional[List[str]]] = {},
    ):
        if attr.upper() == "NORM":
            check_normalizer(nlp)

        self.phrase_matcher = EDSPhraseMatcher(vocab=nlp.vocab, attr=attr)
        self.phrase_matcher.build_patterns(nlp=nlp, terms=terms)

        self.regex_matcher = RegexMatcher(attr=attr)
        self.regex_matcher.build_patterns(regex=regex)
        self.attributes = attributes

        self.on_ents_only = on_ents_only

        if span_getter is None and on_ents_only is None:
            on_ents_only = True

        if on_ents_only:
            assert span_getter is None or on_ents_only is True, (
                "Cannot use both `span_getter` and `on_ents_only` as a span selection "
                "argument."
            )
            assert isinstance(on_ents_only, (list, str, set, bool)), (
                "The `on_ents_only` argument should be a "
                "string, a bool, a list or a set of string"
            )
            span_getter = "ents" if on_ents_only is True else on_ents_only
        else:
            span_getter = "ents"
        self.explain = explain
        super().__init__(
            nlp=nlp,
            name=name,
            span_getter=validate_span_getter(span_getter),
        )

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
            sents = set([ent.sent for ent in self.get_spans(doc)])

            match_iterator = (
                (
                    *self.phrase_matcher(s, as_spans=True),
                    *self.regex_matcher(s, as_spans=True),
                )
                for s in sents
            )

            matches = chain.from_iterable(match_iterator)

        else:
            matches = (
                *self.phrase_matcher(doc, as_spans=True),
                *self.regex_matcher(doc, as_spans=True),
            )

        return list(matches)

    def process(self, doc: Doc) -> BaseQualifierResults:
        raise NotImplementedError

    def __call__(self, doc: Doc) -> Doc:
        results = self.process(doc)
        raise NotImplementedError(f"{type(results)} should be used to tag the document")
