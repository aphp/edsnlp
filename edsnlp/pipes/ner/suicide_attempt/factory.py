from edsnlp.core import registry
from edsnlp.pipes.ner.suicide_attempt.suicide_attempt import SuicideAttemptMatcher

create_component = registry.factory.register(
    "eds.suicide_attempt",
    assigns=["doc.ents", "doc.spans"],
)(SuicideAttemptMatcher)
