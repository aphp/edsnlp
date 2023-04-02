from edsnlp.core import registry

from .diabetes import DiabetesMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="diabetes",
    span_setter={"ents": True, "diabetes": True},
)

create_component = registry.factory.register(
    "eds.diabetes",
    assigns=["doc.ents", "doc.spans"],
)(DiabetesMatcher)
