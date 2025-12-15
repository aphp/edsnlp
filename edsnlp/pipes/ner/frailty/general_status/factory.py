from edsnlp.core import registry

from .general_status import GeneralStatusMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="general_status",
    label="general_status",
    span_setter={"ents": True, "general_status": True},
)

create_component = registry.factory.register(
    "eds.general_status",
    assigns=["doc.ents", "doc.spans"],
)(GeneralStatusMatcher)
