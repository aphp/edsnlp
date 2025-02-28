from edsnlp.core import registry
from edsnlp.pipes.qualifiers.contextual.contextual import ContextualQualifier

create_component = registry.factory.register(
    "eds.contextual_qualifier",
    assigns=["doc.spans"],
)(ContextualQualifier)
