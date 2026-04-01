from edsnlp.core import registry

from .functional_status import FunctionalStatusMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="functional_status",
    label="functional_status",
    span_setter={"ents": True, "functional_status": True},
)

create_component = registry.factory.register(
    "eds.functional_status",
    assigns=["doc.ents", "doc.spans"],
)(FunctionalStatusMatcher)
