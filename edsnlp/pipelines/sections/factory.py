from typing import Dict, List, Optional, Any

from spacy.language import Language

from . import Sections, terms

default_config = dict(
    sections=terms.sections,
    add_patterns=True,
    attr="NORM",
    fuzzy=False,
    fuzzy_kwargs=None,
)


# noinspection PyUnusedLocal
@Language.factory("sections", default_config=default_config)
def create_sections_component(
    nlp: Language,
    name: str,
    sections: Dict[str, List[str]],
    add_patterns: bool,
    attr: str,
    fuzzy: bool,
    fuzzy_kwargs: Optional[Dict[str, Any]],
):
    return Sections(
        nlp,
        sections=sections,
        add_patterns=add_patterns,
        attr=attr,
        fuzzy=fuzzy,
        fuzzy_kwargs=fuzzy_kwargs,
    )
