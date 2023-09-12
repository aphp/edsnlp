from itertools import chain
from typing import Any, Dict, List, Optional

from spacy.language import Language
from spacy.tokens import Doc, Span
from typing_extensions import Literal

from edsnlp.matchers.phrase import EDSPhraseMatcher
from edsnlp.matchers.regex import RegexMatcher
from edsnlp.matchers.simstring import SimstringMatcher
from edsnlp.matchers.utils import Patterns
from edsnlp.pipelines.base import BaseNERComponent, SpanSetterArg


class TerminologyMatcher(BaseNERComponent):
    r"""
    EDS-NLP simplifies the terminology matching process by exposing a `eds.terminology`
    pipeline that can match on terms or regular expressions.

    The terminology matcher is very similar to the
    [generic matcher][edsnlp.pipelines.core.matcher.factory.create_component],
    although the use case differs slightly. The generic matcher is designed to extract
    any entity, while the terminology matcher is specifically tailored towards high
    volume terminologies.

    There are some key differences:

    1. It labels every matched entity to the same value, provided to the pipeline
    2. The keys provided in the `regex` and `terms` dictionaries are used as the
       `kb_id_` of the entity, which handles fine-grained labelling

    For instance, a terminology matcher could detect every drug mention under the
    top-level label `drug`, and link each individual mention to a given drug through
    its `kb_id_` attribute.

    Examples
    --------
    Let us redefine the pipeline :

    ```python
    import spacy

    nlp = spacy.blank("eds")

    terms = dict(
        covid=["coronavirus", "covid19"],  # (1)
        flu=["grippe saisonnière"],  # (2)
    )

    regex = dict(
        covid=r"coronavirus|covid[-\s]?19|sars[-\s]cov[-\s]2",  # (3)
    )

    nlp.add_pipe(
        "eds.terminology",
        config=dict(
            label="disease",
            terms=terms,
            regex=regex,
            attr="LOWER",
        ),
    )
    ```

    1. Every key in the `terms` dictionary is mapped to a concept.
    2. The `eds.matcher` pipeline expects a list of expressions, or a single expression.
    3. We can also define regular expression patterns.

    This snippet is complete, and should run as is.

    Parameters
    ----------
    nlp : Language
        The pipeline object
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
    term_matcher: Literal["exact", "simstring"]
        The matcher to use for matching phrases ?
        One of (exact, simstring)
    term_matcher_config: Dict[str,Any]
        Parameters of the matcher class
    label: str
        Label name to use for the `Span` object and the extension
    span_setter : SpanSetterArg
        How to set matches on the doc

    Patterns, be they `terms` or `regex`, are defined as dictionaries where keys become
    the `kb_id_` of the extracted entities. Dictionary values are either a single
    expression or a list of expressions that match the concept (see [example](#usage)).

    Authors and citation
    --------------------
    The `eds.terminology` pipeline was developed by AP-HP's Data Science team.
    """

    def __init__(
        self,
        nlp: Language,
        name: Optional[str] = None,
        *,
        terms: Optional[Patterns] = None,
        regex: Optional[Patterns] = None,
        attr: str = "TEXT",
        ignore_excluded: bool = False,
        ignore_space_tokens: bool = False,
        term_matcher: Literal["exact", "simstring"] = "exact",
        term_matcher_config: Dict[str, Any] = None,
        label,
        span_setter: SpanSetterArg = {"ents": True},
    ):
        self.label = label

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
                **(term_matcher_config or {}),
            )
        elif term_matcher == "simstring":
            self.phrase_matcher = SimstringMatcher(
                vocab=self.nlp.vocab,
                attr=attr,
                ignore_excluded=ignore_excluded,
                ignore_space_tokens=ignore_space_tokens,
                **(term_matcher_config or {}),
            )
        else:
            raise ValueError(
                f"Algorithm {repr(term_matcher)} does not belong to"
                f" known matchers [exact, simstring]."
            )

        self.regex_matcher = RegexMatcher(
            attr=attr,
            ignore_excluded=ignore_excluded,
            ignore_space_tokens=ignore_space_tokens,
        )

        self.phrase_matcher.build_patterns(nlp=nlp, terms=terms, progress=True)
        self.regex_matcher.build_patterns(regex=regex)

        self.set_extensions()

    def set_extensions(self) -> None:
        super().set_extensions()
        if not Span.has_extension(self.label):
            Span.set_extension(self.label, default=None)

    def process(self, doc: Doc) -> List[Span]:
        """
        Find matching spans in doc.

        Post-process matches to account for terminology.

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

        for match in chain(matches, regex_matches):
            span = Span(
                doc=match.doc,
                start=match.start,
                end=match.end,
                label=self.label,
                kb_id=match.label,
            )
            span._.set(self.label, match.label_)
            yield span

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
