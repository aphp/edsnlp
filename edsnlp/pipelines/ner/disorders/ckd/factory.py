from edsnlp import registry

from .ckd import CKDMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    label="ckd",
    span_setter={"ents": True, "ckd": True},
)

create_component = registry.factory.register(
    "eds.ckd",
    assigns=["doc.ents", "doc.spans"],
    deprecated=["eds.CKD"],
)(CKDMatcher)
