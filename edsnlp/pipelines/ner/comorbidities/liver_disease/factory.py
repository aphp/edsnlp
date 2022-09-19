from typing import Any, Dict, Optional

from spacy.language import Language

from .liver_disease import LiverDisease

DEFAULT_CONFIG = dict(patterns=None)


@Language.factory(
    "eds.comorbidities.liver_disease",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return LiverDisease(nlp, patterns=patterns)
