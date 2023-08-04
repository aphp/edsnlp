from typing import Any, Dict, Optional

from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .CKD import CKD

DEFAULT_CONFIG = dict(patterns=None)


@deprecated_factory(
    "eds.CKD",
    "eds.ckd",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
@Language.factory(
    "eds.ckd",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return CKD(nlp, name=name, patterns=patterns)
