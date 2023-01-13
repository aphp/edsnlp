from typing import List, Optional

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from . import SentenceSegmenter

DEFAULT_CONFIG = dict(
    punct_chars=None,
    ignore_excluded=True,
    use_endlines=None,
    split_on_newlines="with_capitalized",
    split_on_bullets=False,
)


@deprecated_factory(
    "sentences",
    "eds.sentences",
    default_config=DEFAULT_CONFIG,
    assigns=["token.is_sent_start"],
)
@Language.factory(
    "eds.sentences",
    default_config=DEFAULT_CONFIG,
    assigns=["token.is_sent_start"],
)
def create_component(
    nlp: Language,
    name: str,
    punct_chars: Optional[List[str]],
    use_endlines: Optional[bool],
    ignore_excluded: bool,
    split_on_newlines: Optional[str],
    split_on_bullets: Optional[bool],
):
    return SentenceSegmenter(
        nlp.vocab,
        punct_chars=punct_chars,
        use_endlines=use_endlines,
        ignore_excluded=ignore_excluded,
        split_on_newlines=split_on_newlines,
    )
