from typing import Any, Dict, Optional

from spacy.language import Language

from .solid_tumor import SolidTumor

DEFAULT_CONFIG = dict(
    patterns=None,
    use_tnm=False,
)


@Language.factory(
    "eds.solid_tumor",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    patterns: Optional[Dict[str, Any]],
    use_tnm: bool,
):
    return SolidTumor(nlp, name, patterns=patterns, use_tnm=use_tnm)
