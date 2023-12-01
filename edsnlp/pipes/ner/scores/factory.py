from edsnlp.core import registry
from edsnlp.pipes.ner.scores.base_score import SimpleScoreMatcher

DEFAULT_CONFIG = dict(
    regex=None,
    attr="NORM",
    value_extract=None,
    score_normalization=None,
    window=7,
    ignore_excluded=False,
    ignore_space_tokens=False,
    flags=0,
    span_setter={"ents": True},
)

create_component = registry.factory.register(
    "eds.score",
    assigns=["doc.ents", "doc.spans"],
    deprecated=["score"],
)(SimpleScoreMatcher)
