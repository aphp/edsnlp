from typing import Dict, Union

from spacy.language import Language

from edsnlp.pipelines.core.terminology import TerminologyMatcher

from . import patterns

DEFAULT_CONFIG = dict(
    attr="NORM",
    ignore_excluded=False,
)


@Language.factory("eds.cim10", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    attr: Union[str, Dict[str, str]],
    ignore_excluded: bool,
):

    return TerminologyMatcher(
        nlp,
        label="cim10",
        regex=None,
        terms=patterns.get_patterns(),
        attr=attr,
        ignore_excluded=ignore_excluded,
    )
