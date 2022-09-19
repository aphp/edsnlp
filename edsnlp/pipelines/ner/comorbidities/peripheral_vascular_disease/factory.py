from typing import Any, Dict, Optional

from spacy.language import Language

from .peripheral_vascular_disease import PeripheralVascularDisease

DEFAULT_CONFIG = dict(patterns=None)


@Language.factory(
    "eds.comorbidities.peripheral_vascular_disease",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return PeripheralVascularDisease(nlp, patterns=patterns)
