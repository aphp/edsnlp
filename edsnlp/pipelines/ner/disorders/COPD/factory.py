from typing import Any, Dict, Optional

from spacy.language import Language

from .COPD import COPD

DEFAULT_CONFIG = dict(patterns=None)


@Language.factory(
    "eds.COPD",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return COPD(nlp, patterns=patterns)
