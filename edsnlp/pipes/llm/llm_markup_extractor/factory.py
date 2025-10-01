from edsnlp import registry

from .llm_markup_extractor import LlmMarkupExtractor

create_component = registry.factory.register(
    "eds.llm_markup_extractor",
    assigns=["doc.ents", "doc.spans"],
)(LlmMarkupExtractor)
