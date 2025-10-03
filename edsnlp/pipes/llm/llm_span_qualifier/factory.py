from edsnlp import registry

from .llm_span_qualifier import LlmSpanQualifier

create_component = registry.factory.register(
    "eds.llm_span_qualifier",
    assigns=["doc.ents", "doc.spans"],
)(LlmSpanQualifier)
