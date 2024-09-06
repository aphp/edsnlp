from edsnlp import registry

from .sentences import SentenceSegmenter

create_component = registry.factory.register(
    "eds.sentences",
    assigns=["token.is_sent_start"],
    deprecated=["sentences"],
)(SentenceSegmenter)
