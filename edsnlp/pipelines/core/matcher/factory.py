from typing import Dict, List, Optional, Union

from spacy.language import Language

from edsnlp.pipelines.core.matcher import GenericMatcher, FlashTextComponent
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    terms=None,
    regex=None,
    attr="TEXT",
    ignore_excluded=False,
)


@deprecated_factory("matcher", "eds.matcher", default_config=DEFAULT_CONFIG)
@Language.factory("eds.matcher", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    terms: Optional[Dict[str, Union[str, List[str]]]],
    attr: Union[str, Dict[str, str]],
    regex: Optional[Dict[str, Union[str, List[str]]]],
    ignore_excluded: bool,
):
    assert not (terms is None and regex is None)

    if terms is None:
        terms = dict()
    if regex is None:
        regex = dict()

    return GenericMatcher(
        nlp,
        terms=terms,
        attr=attr,
        regex=regex,
        ignore_excluded=ignore_excluded,
    )


DEFAULT_CONFIG = dict(
    terms=None,
    regex=None,
    attr="TEXT",
    ignore_excluded=False,
    max_cost=0,
)
@deprecated_factory("matcherbis", "flashtext.matcher", default_config=DEFAULT_CONFIG)
@Language.factory("flashtext.matcher", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    terms: Optional[Dict[str, Union[str, List[str]]]],
    attr: Union[str, Dict[str, str]],
    regex: Optional[Dict[str, Union[str, List[str]]]],
    ignore_excluded: bool,
    max_cost: int,
):
    assert not (terms is None and regex is None)

    if terms is None:
        terms = dict()
    if regex is None:
        regex = dict()

    return FlashTextComponent(
        nlp,
        terms=terms,
        attr=attr,
        regex=regex,
        ignore_excluded=ignore_excluded,
        max_cost = max_cost
    )