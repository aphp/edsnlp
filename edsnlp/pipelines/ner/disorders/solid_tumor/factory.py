from edsnlp.core import registry

from .patterns import default_patterns
from .solid_tumor import SolidTumorMatcher

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    use_tnm=False,
    label="solid_tumor",
    span_setter={"ents": True, "solid_tumor": True},
)

create_component = registry.factory.register(
    "eds.solid_tumor",
    assigns=["doc.ents", "doc.spans"],
)(SolidTumorMatcher)
