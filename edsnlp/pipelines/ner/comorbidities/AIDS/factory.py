from typing import Any, Dict, Optional

from spacy.language import Language

from .AIDS import AIDS

DEFAULT_CONFIG = dict(patterns=None)


@Language.factory(
    "eds.comorbidities.AIDS",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return AIDS(nlp, patterns=patterns)
