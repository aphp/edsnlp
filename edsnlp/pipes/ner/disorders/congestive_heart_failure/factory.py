from edsnlp.core import registry

from .congestive_heart_failure import CongestiveHeartFailureMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="congestive_heart_failure",
    span_setter={"ents": True, "congestive_heart_failure": True},
)

create_component = registry.factory.register(
    "eds.congestive_heart_failure",
    assigns=["doc.ents", "doc.spans"],
)(CongestiveHeartFailureMatcher)
