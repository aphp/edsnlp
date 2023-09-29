from edsnlp import registry

from .span_pooler import SpanPooler

create_component = registry.factory.register(
    "eds.span_pooler",
    assigns=[],
    deprecated=[],
)(SpanPooler)
