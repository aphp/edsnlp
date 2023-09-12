from typing import Any, Dict, List, Optional

from spacy.language import Language
from spacy.tokens import Doc, Span
from typing_extensions import Literal

from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.matchers.simstring import SimstringMatcher
from edsnlp.matchers.utils import Patterns
from edsnlp.pipelines.base import BaseNERComponent, SpanSetterArg


class GenericMatcher(BaseNERComponent):
    r"""
    EDS-NLP simplifies the matching process by exposing a `eds.matcher` component
    that can match on terms or regular expressions.

    Examples
    --------
    Let us redefine the pipeline :

    ```python
    import spacy

    nlp = spacy.blank("eds")

    terms = dict(
        covid=["coronavirus", "covid19"],  # (1)
        patient="patient",  # (2)
    )

    regex = dict(
        covid=r"coronavirus|covid[-\s]?19|sars[-\s]cov[-\s]2",  # (3)
    )

    nlp.add_pipe(
        "eds.matcher",
        config=dict(
            terms=terms,
            regex=regex,
            attr="LOWER",
            term_matcher="exact",
            term_matcher_config={},
        ),
    )
    ```

    1. Every key in the `terms` dictionary is mapped to a concept.
    2. The `eds.matcher` pipeline expects a list of expressions, or a single expression.
    3. We can also define regular expression patterns.

    This snippet is complete, and should run as is.

    Patterns, be they `terms` or `regex`, are defined as dictionaries where keys become
     the label of the extracted entities. Dictionary values are either a single
     expression or a list of expressions that match the concept.

    Parameters
    ----------
    nlp : Language
        The pipeline object.
    name: str
        The name of the component.
    terms : Optional[Patterns]
        A dictionary of terms.
    regex : Optional[Patterns]
        A dictionary of regular expressions.
    attr : str
        The default attribute to use for matching.
        Can be overridden using the `terms` and `regex` configurations.
    ignore_excluded : bool
        Whether to skip excluded tokens (requires an upstream
        pipeline to mark excluded tokens).
    ignore_space_tokens: bool
        Whether to skip space tokens during matching.

        You won't be able to match on newlines if this is enabled and
        the "spaces"/"newline" option of `eds.normalizer` is enabled (by default).
    term_matcher : Literal["exact", "simstring"]
        The matcher to use for matching phrases ?
        One of (exact, simstring)
    term_matcher_config : Dict[str,Any]
        Parameters of the matcher class
    span_setter : SpanSetterArg
        How to set the spans in the doc.

    Authors and citation
    --------------------
    The `eds.matcher` pipeline was developed by AP-HP's Data Science team.
    """

    def __init__(
        self,
        nlp: Language,
        name: Optional[str] = "eds.matcher",
        *,
        terms: Optional[Patterns] = None,
        regex: Optional[Patterns] = None,
        attr: str = "TEXT",
        ignore_excluded: bool = False,
        ignore_space_tokens: bool = False,
        term_matcher: Literal["exact", "simstring"] = "exact",
        term_matcher_config: Dict[str, Any] = {},
        span_setter: SpanSetterArg = {"ents": True},
    ):
        super().__init__(nlp=nlp, name=name, span_setter=span_setter)

        if terms is None and regex is None:
            raise ValueError(
                "You must provide either `terms` or `regex` to the matcher."
            )

        terms = terms or {}
        regex = regex or {}

        self.attr = attr

        if term_matcher == "exact":
            self.phrase_matcher = EDSPhraseMatcher(
                self.nlp.vocab,
                attr=attr,
                ignore_excluded=ignore_excluded,
                ignore_space_tokens=ignore_space_tokens,
                **term_matcher_config,
            )
        elif term_matcher == "simstring":
            self.phrase_matcher = SimstringMatcher(
                self.nlp.vocab,
                attr=attr,
                ignore_excluded=ignore_excluded,
                ignore_space_tokens=ignore_space_tokens,
                **term_matcher_config,
            )
        else:
            raise ValueError(
                f"Algorithm {repr(term_matcher)} does not belong to"
                f" known matcher [exact, simstring]."
            )

        self.regex_matcher = RegexMatcher(
            attr=attr,
            ignore_excluded=ignore_excluded,
            ignore_space_tokens=ignore_space_tokens,
        )

        self.phrase_matcher.build_patterns(nlp=nlp, terms=terms)
        self.regex_matcher.build_patterns(regex=regex)

        self.set_extensions()

    def process(self, doc: Doc) -> List[Span]:
        """
        Find matching spans in doc.

        Parameters
        ----------
        doc:
            spaCy Doc object.

        Returns
        -------
        spans:
            List of Spans returned by the matchers.
        """

        matches = self.phrase_matcher(doc, as_spans=True)
        regex_matches = self.regex_matcher(doc, as_spans=True)

        spans = list(matches) + list(regex_matches)

        return spans

    def __call__(self, doc: Doc) -> Doc:
        """
        Adds spans to document.

        Parameters
        ----------
        doc:
            spaCy Doc object

        Returns
        -------
        doc:
            spaCy Doc object, annotated for extracted terms.
        """
        matches = self.process(doc)

        self.set_spans(doc, matches)

        return doc
