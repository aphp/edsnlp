from edsnlp import registry

from .transformer import Transformer

create_component = registry.factory.register(
    "eds.transformer",
    assigns=[],
    deprecated=[],
)(Transformer)
