from typing import Any, Dict, Optional

from spacy.language import Language

from .hemiplegia import Hemiplegia

DEFAULT_CONFIG = dict(patterns=None)


@Language.factory(
    "eds.hemiplegia",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return Hemiplegia(nlp, name=name, patterns=patterns)
