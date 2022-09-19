from typing import Any, Dict, Optional

from spacy.language import Language

from .CKD import CKD

DEFAULT_CONFIG = dict(patterns=None)


@Language.factory(
    "eds.comorbidities.CKD",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return CKD(nlp, patterns=patterns)
