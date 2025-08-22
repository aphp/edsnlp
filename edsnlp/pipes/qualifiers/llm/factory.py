from edsnlp.core import registry

from .llm_qualifier import LLMSpanClassifier

create_component = registry.factory.register(
    "eds.llm_span_qualifier",
)(LLMSpanClassifier)
