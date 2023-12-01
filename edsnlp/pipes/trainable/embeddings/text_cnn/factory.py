from edsnlp import registry

from .text_cnn import TextCnnEncoder

create_component = registry.factory.register(
    "eds.text_cnn",
    assigns=[],
    deprecated=[],
)(TextCnnEncoder)
