from edsnlp.core import registry

from .endlines import EndLinesMatcher

DEFAULT_CONFIG = dict(
    model_path=None,
)

create_component = registry.factory.register(
    "eds.endlines",
    assigns=["doc.ents", "doc.spans"],
    deprecated=["spaces"],
)(EndLinesMatcher)
