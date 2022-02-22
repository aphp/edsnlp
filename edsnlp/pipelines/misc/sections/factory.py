from typing import Dict, List

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from . import Sections, patterns

DEFAULT_CONFIG = dict(
    sections=patterns.sections,
)


@deprecated_factory("sections", "eds.sections", default_config=DEFAULT_CONFIG)
@Language.factory("eds.sections", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    sections: Dict[str, List[str]],
    add_patterns: bool = True,
    attr: str = "NORM",
    ignore_excluded: bool = True,
):
    return Sections(
        nlp,
        sections=sections,
        add_patterns=add_patterns,
        attr=attr,
        ignore_excluded=ignore_excluded,
    )
