from typing import Dict, Union

from spacy.language import Language

from edsnlp.pipelines.core.matcher import GenericMatcher

from . import patterns

DEFAULT_CONFIG = dict(
    attr="LOWER",
    ignore_excluded=False,
)


@Language.factory(
    "eds.covid",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str = "eds.covid",
    attr: Union[str, Dict[str, str]] = "LOWER",
    ignore_excluded: bool = False,
):
    """
    Create a factory that returns new GenericMatcher with patterns for covid

    Parameters
    ----------
    nlp: Language
        spaCy `Language` object.
    name: str
        The name of the pipe
    attr: Union[str, Dict[str, str]]
        Attribute to match on, eg `TEXT`, `NORM`, etc.
    ignore_excluded: bool
        Whether to skip excluded tokens during matching.

    Returns
    -------
    GenericMatcher
    """

    return GenericMatcher(
        nlp,
        terms=None,
        regex=dict(covid=patterns.pattern),
        attr=attr,
        ignore_excluded=ignore_excluded,
    )
