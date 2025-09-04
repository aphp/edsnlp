import re
from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

from pydantic import NonNegativeInt
from spacy.tokens import Doc, Span

from edsnlp.core import PipelineProtocol
from edsnlp.matchers.utils import Patterns
from edsnlp.pipes.base import (
    BaseSpanAttributeClassifierComponent,
    SpanGetterArg,
)
from edsnlp.pipes.core.matcher.matcher import GenericMatcher
from edsnlp.utils.span_getters import (
    get_spans,
    make_span_context_getter,
    validate_span_getter,
)


@dataclass
class ClassPatternsContext:
    """
    A data class to hold pattern matching context.

    Parameters
    ----------
    terms : Optional[Patterns]
        Terms to match.
    regex : Optional[Patterns]
        Regular expressions to match.
    context_words : Union[NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt]]
        Number of words to consider as context.
    context_sents : Union[NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt]]
        Number of sentences to consider as context.
    attr : str
        Attribute to match on.
    regex_flags : Union[re.RegexFlag, int]
        Flags for regular expressions.
    ignore_excluded : bool
        Whether to ignore excluded tokens.
    ignore_space_tokens : bool
        Whether to ignore space tokens.
    """

    terms: Optional[Patterns] = None
    regex: Optional[Patterns] = None
    context_words: Union[NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt]] = 0
    context_sents: Union[NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt]] = 1
    attr: str = "TEXT"
    regex_flags: Union[re.RegexFlag, int] = 0
    ignore_excluded: bool = False
    ignore_space_tokens: bool = False


@dataclass
class ClassMatcherContext:
    """
    A data class to hold matcher context configuration.

    Parameters
    ----------
    name : str
        The name of the matcher.
    value : Union[str, bool, int]
        The value to set when a match is found.
    matcher : GenericMatcher
        The matcher object.
    context_words : Union[NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt]]
        Number of words to consider as context.
    context_sents : Union[NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt]]
        Number of sentences to consider as context.
    """

    name: str
    value: Union[str, bool, int]
    matcher: GenericMatcher
    context_words: Union[NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt]]
    context_sents: Union[NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt]]


class ContextualQualifier(BaseSpanAttributeClassifierComponent):
    """
    The `eds.contextual_qualifier` pipeline component
    qualifies spans based on contextual information.

    Parameters
    ----------
    nlp : PipelineProtocol
        The spaCy pipeline object.
    name : Optional[str], default="contextual_qualifier"
        The name of the component.
    span_getter : SpanGetterArg
        The function or callable to get spans from the document.
    patterns : Dict[str, Dict[Union[str, int], ClassPatternsContext]]
        A dictionary of patterns to match in the text. Each pattern dictionary should
        follow the structure of the `ClassPatternsContext` data class.

        ??? note "`ClassPatternsContext`"
            ::: edsnlp.pipes.qualifiers.contextual.contextual.ClassPatternsContext
                options:
                    heading_level: 1
                    only_parameters: "no-header"
                    skip_parameters: []
                    show_source: false
                    show_toc: false

    """

    def __init__(
        self,
        nlp: PipelineProtocol,
        name: Optional[str] = "contextual_qualifier",
        *,
        span_getter: SpanGetterArg,
        patterns: Dict[str, Dict[Union[str, int], ClassPatternsContext]],
    ):
        """
        Initialize the ContextualQualifier.

        Parameters
        ----------
        nlp : PipelineProtocol
            The NLP pipeline object.
        name : Optional[str], default="contextual_qualifier"
            The name of the qualifier.
        span_getter : SpanGetterArg
            The span getter argument to identify spans to qualify.
        patterns : Dict[str, Dict[Union[str, int], ClassPatternsContext]]
            A dictionary of patterns to match in the text.
        """
        self.span_getter = span_getter
        self.named_matchers = list()  # Will contain all the named matchers

        for pattern_name, named_dict in patterns.items():
            for value, class_patterns in named_dict.items():
                if isinstance(class_patterns, dict):
                    class_patterns = ClassPatternsContext(**class_patterns)

                name_value_str = str(pattern_name) + "_" + str(value)
                name_value_matcher = GenericMatcher(
                    nlp=nlp,
                    terms=class_patterns.terms,
                    regex=class_patterns.regex,
                    attr=class_patterns.attr,
                    ignore_excluded=class_patterns.ignore_excluded,
                    ignore_space_tokens=class_patterns.ignore_space_tokens,
                    span_setter=name_value_str,
                )

                self.named_matchers.append(
                    ClassMatcherContext(
                        name=pattern_name,
                        value=value,
                        matcher=name_value_matcher,
                        context_words=class_patterns.context_words,
                        context_sents=class_patterns.context_sents,
                    )
                )

        self.set_extensions()

        super().__init__(
            nlp=nlp,
            name=name,
            span_getter=validate_span_getter(span_getter),
        )

    def set_extensions(self) -> None:
        """
        Sets custom extensions on the Span object for each context name.
        """
        for named_matcher in self.named_matchers:
            if not Span.has_extension(named_matcher.name):
                Span.set_extension(named_matcher.name, default=None)

    def get_matches(self, context: Span) -> List[Span]:
        """
        Extracts matches from the context span.

        Parameters
        ----------
        context : Span
            The span context to look for a match.

        Returns
        -------
        List[Span]
            List of detected spans.
        """
        match_iterator = (
            *self.phrase_matcher(context, as_spans=True),
            *self.regex_matcher(context, as_spans=True),
        )

        matches = chain.from_iterable(match_iterator)

        return list(matches)

    def __call__(self, doc: Doc) -> Doc:
        """
        Processes the document, qualifying spans based on contextual information.

        Parameters
        ----------
        doc : Doc
            The spaCy document to process.

        Returns
        -------
        Doc
            The processed document with qualified spans.
        """
        for matcher in self.named_matchers:
            span_context_getter = make_span_context_getter(
                context_words=matcher.context_words, context_sents=matcher.context_sents
            )
            for ent in get_spans(doc, self.span_getter):
                context = span_context_getter(ent)

                matches = matcher.matcher.process(context)

                if len(matches) > 0:
                    ent._.set(matcher.name, matcher.value)

        return doc
