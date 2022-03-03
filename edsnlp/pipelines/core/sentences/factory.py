from typing import List, Optional

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from . import SentenceSegmenter

DEFAULT_CONFIG = dict(
    punct_chars=None,
    use_endlines=True,
)


@deprecated_factory("sentences", "eds.sentences", default_config=DEFAULT_CONFIG)
@Language.factory("eds.sentences", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    punct_chars: Optional[List[str]],
    use_endlines: bool,
):
    return SentenceSegmenter(
        punct_chars=punct_chars,
        use_endlines=use_endlines,
    )
