from edsnlp.core import registry

from .nutrition import NutritionMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="nutrition",
    label="nutrition",
    span_setter={"ents": True, "nutrition": True},
)

create_component = registry.factory.register(
    "eds.nutrition",
    assigns=["doc.ents", "doc.spans"],
)(NutritionMatcher)
