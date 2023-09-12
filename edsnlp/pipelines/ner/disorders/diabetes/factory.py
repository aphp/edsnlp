from spacy import Language

from .diabetes import DiabetesMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="diabetes",
    span_setter={"ents": True, "diabetes": True},
)

create_component = Language.factory(
    "eds.diabetes",
    assigns=["doc.ents", "doc.spans"],
)(DiabetesMatcher)
