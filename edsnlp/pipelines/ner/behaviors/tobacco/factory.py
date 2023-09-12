from spacy import Language

from .patterns import default_patterns
from .tobacco import TobaccoMatcher

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="tobacco",
    span_setter={"ents": True, "tobacco": True},
)

create_component = Language.factory(
    "eds.tobacco",
    assigns=["doc.ents", "doc.spans"],
)(TobaccoMatcher)
