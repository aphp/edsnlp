from typing import Any, Dict, Optional

from spacy.language import Language

from .myasthenia import Myasthenia

DEFAULT_CONFIG = dict(patterns=None)


@Language.factory(
    "eds.comorbidities.myasthenia",
    default_config=DEFAULT_CONFIG,
    assigns=["doc.ents", "doc.spans"],
)
def create_component(
    nlp: Language,
    name: str,
    patterns: Optional[Dict[str, Any]],
):
    return Myasthenia(nlp, patterns=patterns)
