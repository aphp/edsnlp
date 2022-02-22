from typing import List, Optional

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from . import SentenceSegmenter
from .terms import punctuation

DEFAULT_CONFIG = dict(punct_chars=punctuation)


@deprecated_factory("sentences", "eds.sentences", default_config=DEFAULT_CONFIG)
@Language.factory("eds.sentences", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    punct_chars: Optional[List[str]] = None,
    use_endlines: bool = True,
):
    return SentenceSegmenter(punct_chars, use_endlines=use_endlines)
