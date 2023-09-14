from edsnlp.core import registry

from .patterns import default_enabled
from .pollution import PollutionTagger

DEFAULT_CONFIG = dict(
    pollution=default_enabled,
)

create_component = registry.factory.register(
    "eds.pollution",
    assigns=["doc.spans"],
    deprecated=["pollution"],
)(PollutionTagger)
