from edsnlp.core import registry

from .geriatric_assessment import GAMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="geriatric_assessment",
    label="geriatric_assessment",
    span_setter={"ents": True, "geriatric_assessment": True},
)

create_component = registry.factory.register(
    "eds.geriatric_assessment",
    assigns=["doc.ents", "doc.spans"],
)(GAMatcher)
