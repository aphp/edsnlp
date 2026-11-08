from edsnlp.core import registry

from .nutritional_status import NutritionMatcher
from .patterns import default_patterns

DEFAULT_CONFIG = dict(
    patterns=default_patterns,
    domain="nutritional_status",
    label="nutritional_status",
    span_setter={"ents": True, "nutritional_status": True},
)

create_component = registry.factory.register(
    "eds.nutritional_status",
    assigns=["doc.ents", "doc.spans"],
)(NutritionMatcher)
