from typing import Dict, Any, List, Optional

from spacy.language import Language

from edsnlp.rules.advanced import AdvancedRegex
from edsnlp.rules.generic import GenericMatcher
from edsnlp.rules.normalise import Normaliser
from edsnlp.rules.pollution import Pollution, terms as pollution_terms
from edsnlp.rules.quickumls import QuickUMLSComponent
from edsnlp.rules.sections import Sections, terms as section_terms
from edsnlp.rules.sentences import SentenceSegmenter

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
        add_patterns: bool = True,
        attr: str = 'NORM',
):
    return Sections(nlp, sections=sections, add_patterns=add_patterns, attr=attr)


# noinspection PyUnusedLocal
@Language.factory("quickumls")
def create_quickumls_component(
        nlp: Language,
        name: str,
        distribution: str,
):
    return QuickUMLSComponent(nlp, distribution=distribution)


# noinspection PyUnusedLocal
@Language.factory("sentences")
def create_sentences_component(
        nlp: Language,
        name: str,
        punct_chars: Optional[List[str]] = None,
):
    return SentenceSegmenter(punct_chars)


# noinspection PyUnusedLocal
@Language.factory("matcher")
def create_matcher_component(
        nlp: Language,
        name: str,
        terms: Optional[Dict[str, List[str]]] = None,
        attr: str = 'TEXT',
        regex: Optional[Dict[str, List[str]]] = None,
        fuzzy: bool = False,
        fuzzy_kwargs: Optional[Dict[str, Any]] = None,
        filter_matches: bool = True,
        on_ents_only: bool = False
):
    if terms is None:
        terms = dict()
    if regex is None:
        regex = dict()

    return GenericMatcher(
        nlp,
        terms=terms,
        attr=attr,
        regex=regex,
        fuzzy=fuzzy,
        fuzzy_kwargs=fuzzy_kwargs,
        filter_matches=filter_matches,
        on_ents_only=on_ents_only
    )


@Language.factory("advanced_regex")
def create_adv_regex_component(
        nlp: Language,
        name: str,
        regex_config: Dict[str, Any],
        window: int
):

    return AdvancedRegex(
        nlp,
        regex_config,
        window
    )


# noinspection PyUnusedLocal
@Language.factory("normaliser")
def create_normaliser_component(
        nlp: Language,
        name: str,
        deaccentuate: bool = True,
        lowercase: bool = True,
):
    return Normaliser(deaccentuate=deaccentuate, lowercase=lowercase)
