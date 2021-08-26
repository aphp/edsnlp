from typing import List, Optional

from spacy.language import Language

from . import SentenceSegmenter


# noinspection PyUnusedLocal
@Language.factory("sentences")
def create_sentences_component(
    nlp: Language,
    name: str,
    punct_chars: Optional[List[str]] = None,
):
    return SentenceSegmenter(punct_chars)
