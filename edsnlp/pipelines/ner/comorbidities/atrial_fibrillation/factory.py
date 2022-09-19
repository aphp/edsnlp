from typing import Any, Dict, Optional

from spacy.language import Language

from .atrial_fibrillation import AtrialFibrillation

DEFAULT_CONFIG = dict(patterns=None)


@Language.factory(
    "eds.comorbidities.atrial_fibrillation",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return AtrialFibrillation(nlp, patterns=patterns)
