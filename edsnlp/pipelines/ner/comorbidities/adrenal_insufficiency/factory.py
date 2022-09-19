from typing import Any, Dict, Optional

from spacy.language import Language

from .adrenal_insufficiency import AdrenalInsufficiency

DEFAULT_CONFIG = dict(patterns=None)


@Language.factory(
    "eds.comorbidities.adrenal_insufficiency",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return AdrenalInsufficiency(nlp, patterns=patterns)
