from typing import List, Optional

from spacy.language import Language

from . import SentenceSegmenter
from .terms import punctuation

default_config = dict(punct_chars=punctuation)


# noinspection PyUnusedLocal
@Language.factory("sentences", default_config=default_config)
def create_component(
    nlp: Language,
    name: str,
    punct_chars: Optional[List[str]] = None,
    use_endlines: bool = True,
):
    return SentenceSegmenter(punct_chars, use_endlines=use_endlines)
