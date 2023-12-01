from edsnlp.core import registry

from .lymphoma import LymphomaMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="lymphoma",
    span_setter={"ents": True, "lymphoma": True},
)

create_component = registry.factory.register(
    "eds.lymphoma",
    assigns=["doc.ents", "doc.spans"],
)(LymphomaMatcher)
