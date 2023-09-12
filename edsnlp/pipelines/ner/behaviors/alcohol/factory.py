from spacy import Language

from .alcohol import AlcoholMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="alcohol",
    span_setter={"ents": True, "alcohol": True},
)

create_component = Language.factory(
    "eds.alcohol",
    assigns=["doc.ents", "doc.spans"],
)(AlcoholMatcher)
