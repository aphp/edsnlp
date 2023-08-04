from typing import Any, Dict, Optional

from spacy.language import Language

from .diabetes import Diabetes

DEFAULT_CONFIG = dict(patterns=None)


@Language.factory(
    "eds.diabetes",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return Diabetes(nlp, name=name, patterns=patterns)
