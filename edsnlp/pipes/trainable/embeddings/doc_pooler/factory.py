from edsnlp import registry

from .doc_pooler import DocPooler

create_component = registry.factory.register(
    "eds.doc_pooler",
    assigns=[],
    deprecated=[],
)(DocPooler)
