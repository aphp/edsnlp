from typing import Dict, List, Optional

from spacy.language import Language

from nlptools.rules.pollution import Pollution, terms as pollution_terms
from nlptools.rules.sections import Sections, terms as section_terms
from nlptools.rules.quickumls import QuickUMLSComponent
from nlptools.rules.generic import GenericMatcher

pollution_default_config = dict(
    pollution=pollution_terms.pollution,
)


# noinspection PyUnusedLocal
@Language.factory("pollution", default_config=pollution_default_config)
def create_pollution_component(
        nlp: Language,
        name: str,
        pollution: Dict[str, str],
):
    return Pollution(nlp, pollution=pollution)


sections_default_config = dict(
    sections=section_terms.sections,
)


# noinspection PyUnusedLocal
@Language.factory("sections", default_config=sections_default_config)
def create_sections_component(
        nlp: Language,
        name: str,
        sections: Dict[str, List[str]],
):
    return Sections(nlp, sections=sections)


# noinspection PyUnusedLocal
@Language.factory("quickumls")
def create_quickumls_component(
        nlp: Language,
        name: str,
        distribution: str,
):
    return QuickUMLSComponent(nlp, distribution=distribution)


# noinspection PyUnusedLocal
@Language.factory("matcher")
def create_matcher_component(
        nlp: Language,
        name: str,
        terms: Optional[Dict[str, List[str]]] = None,
        regex: Optional[Dict[str, List[str]]] = None,
        fuzzy: Optional[bool] = False,
):
    if terms is None:
        terms = dict()
    if regex is None:
        regex = dict()
    return GenericMatcher(nlp, terms=terms, regex=regex, fuzzy=fuzzy)
