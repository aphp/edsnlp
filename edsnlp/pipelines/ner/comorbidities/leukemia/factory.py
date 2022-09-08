from typing import Any, List, Optional, Union, Dict

from spacy.language import Language

from .leukemia import Leukemia

DEFAULT_CONFIG = dict(patterns=None)


@Language.factory(
    "eds.comorbidities.leukemia",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return Leukemia(nlp, patterns=patterns)
