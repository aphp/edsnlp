import re
from functools import partial
from typing import Dict, Generator, List, Optional, Union

from spacy import registry
from spacy.language import Language, Vocab
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token

from .utils import ATTRIBUTES, Patterns

PatternDict = Dict[str, Union[str, Dict[str, str]]]


def get_normalized_variant(doclike: Union[Span, Doc]) -> str:
    tokens = [t.text + t.whitespace_ for t in doclike if not t._.excluded]
    variant = "".join(tokens)
    variant = variant.rstrip(" ")
    variant = re.sub(r"\s+", " ", variant)
    return variant


if not Span.has_extension("normalized_variant"):
    Span.set_extension("normalized_variant", getter=get_normalized_variant)


@registry.misc("edsnlp.factories.phrasematcher.v1")
def phrase_matcher_factory(
    attr: str,
    ignore_excluded: bool,
    exclude_newlines: bool,
):
    return partial(
        EDSPhraseMatcher,
        attr=attr,
        ignore_excluded=ignore_excluded,
        exclude_newlines=exclude_newlines,
    )


class EDSPhraseMatcher(object):
    """
    PhraseMatcher that matches "over" excluded tokens.

    Parameters
    ----------
    vocab : Vocab
        spaCy vocabulary to match on.
    attr : str
        Default attribute to match on, by default "TEXT".
        Can be overiden in the `add` method.

        To match on a custom attribute, prepend the attribute name with `_`.
    ignore_excluded : bool, optional
        Whether to ignore excluded tokens, by default True
    exclude_newlines : bool, optional
        Whether to exclude new lines, by default False
    """

    def __init__(
        self,
        vocab: Vocab,
        attr: str = "TEXT",
        ignore_excluded: bool = True,
        exclude_newlines: bool = False,
    ):
        self.matcher = Matcher(vocab, validate=True)
        self.attr = attr
        self.ignore_excluded = ignore_excluded

        self.exclusion_attribute = (
            "excluded_or_space" if exclude_newlines else "excluded"
        )

    @staticmethod
    def get_attr(token: Token, attr: str, custom_attr: bool = False) -> str:
        if custom_attr:
            return getattr(token._, attr)
        else:
            attr = ATTRIBUTES.get(attr)
            return getattr(token, attr)

    def create_pattern(
        self,
        match_pattern: Doc,
        attr: Optional[str] = None,
        ignore_excluded: Optional[bool] = None,
    ) -> List[PatternDict]:
        """
        Create a pattern

        Parameters
        ----------
        match_pattern : Doc
            A spaCy doc object, to use as match model.
        attr : str, optional
            Overwrite attribute to match on.
        ignore_excluded: bool, optional
            Whether to skip excluded tokens.

        Returns
        -------
        List[PatternDict]
            A spaCy rule-based pattern.
        """

        ignore_excluded = ignore_excluded or self.ignore_excluded

        attr = attr or self.attr
        custom_attr = attr.startswith("_")

        if custom_attr:
            attr = attr.lstrip("_").lower()

            pattern = []

            for token in match_pattern:
                pattern.append({"_": {attr: self.get_attr(token, attr, True)}})
                if ignore_excluded and token.whitespace_:
                    # If the token is followed by a whitespace,
                    # we let it match on a pollution
                    pattern.append({"_": {self.exclusion_attribute: True}, "OP": "*"})

            return pattern
        else:
            pattern = []

            for token in match_pattern:
                pattern.append({attr: self.get_attr(token, attr, False)})
                if ignore_excluded and token.whitespace_:
                    # If the token is followed by a whitespace,
                    # we let it match on a pollution
                    pattern.append({"_": {self.exclusion_attribute: True}, "OP": "*"})

            return pattern

    def build_patterns(self, nlp: Language, terms: Patterns):
        """
        Build patterns and adds them for matching.
        Helper function for pipelines using this matcher.

        Parameters
        ----------
        nlp : Language
            The instance of the spaCy language class.
        terms : Patterns
            Dictionary of label/terms, or label/dictionary of terms/attribute.
        """

        if not terms:
            terms = dict()

        for key, expressions in terms.items():
            if isinstance(expressions, dict):
                attr = expressions.get("attr")
                expressions = expressions.get("patterns")
            else:
                attr = None
            if isinstance(expressions, str):
                expressions = [expressions]
            patterns = list(nlp.pipe(expressions))
            self.add(key, patterns, attr)

    def add(
        self,
        key: str,
        patterns: List[Doc],
        attr: Optional[str] = None,
        ignore_excluded: Optional[bool] = None,
    ) -> None:
        """
        Add a pattern.

        Parameters
        ----------
        key : str
            Key of the new/updated pattern.
        patterns : List[str]
            List of patterns to add.
        attr : str, optional
            Overwrite the attribute to match on for this specific pattern.
        ignore_excluded : bool, optional
            Overwrite the parameter for this specific pattern.
        """

        patterns = [
            self.create_pattern(pattern, attr=attr, ignore_excluded=ignore_excluded)
            for pattern in patterns
        ]
        self.matcher.add(key, patterns)

    def remove(
        self,
        key: str,
    ) -> None:
        """
        Remove a pattern.

        Parameters
        ----------
        key : str
            key of the pattern to remove.

        Raises
        ------
        ValueError
            Should the key not be contained in the registry.
        """
        self.matcher.remove(key)

    def __len__(self):
        return len(self.matcher)

    def __call__(
        self,
        doclike: Union[Doc, Span],
        as_spans=False,
    ) -> Generator:
        """
        Performs matching. Yields matches.

        Parameters
        ----------
        doclike:
            spaCy Doc or Span object.
        as_spans:
            Whether to return matches as spans.

        Yields
        -------
        match: Span
            A match.
        """
        if len(self.matcher):
            for match in self.matcher(doclike, as_spans=as_spans):
                yield match
