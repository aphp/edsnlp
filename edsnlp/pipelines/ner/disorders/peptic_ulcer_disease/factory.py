from typing import Any, Dict, Optional

from spacy.language import Language

from .peptic_ulcer_disease import PepticUlcerDisease

DEFAULT_CONFIG = dict(patterns=None)


@Language.factory(
    "eds.peptic_ulcer_disease",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return PepticUlcerDisease(nlp, patterns=patterns)
