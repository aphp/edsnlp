from typing import Any, Dict, List, Optional

from spacy.language import Language

from . import Sections, terms

default_config = dict(
    sections=terms.sections,
)


# noinspection PyUnusedLocal
@Language.factory("sections", default_config=default_config)
def create_component(
    nlp: Language,
    name: str,
    sections: Dict[str, List[str]],
    add_patterns: bool = True,
    attr: str = "NORM",
    fuzzy: bool = False,
    fuzzy_kwargs: Optional[Dict[str, Any]] = None,
):
    return Sections(
        nlp,
        sections=sections,
        add_patterns=add_patterns,
        attr=attr,
        fuzzy=fuzzy,
        fuzzy_kwargs=fuzzy_kwargs,
    )
