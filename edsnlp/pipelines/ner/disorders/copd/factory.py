from edsnlp import registry

from .copd import COPDMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="copd",
    span_setter={"ents": True, "copd": True},
)

create_component = registry.factory.register(
    "eds.copd",
    assigns=["doc.ents", "doc.spans"],
    deprecated=["eds.COPD"],
)(COPDMatcher)
