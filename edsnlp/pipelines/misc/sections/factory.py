from typing import Dict, List

from spacy.language import Language

from . import Sections, patterns

default_config = dict(
    sections=patterns.sections,
)


# noinspection PyUnusedLocal
@Language.factory("sections", default_config=default_config)
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
