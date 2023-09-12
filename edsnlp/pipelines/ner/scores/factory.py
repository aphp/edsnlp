from spacy.language import Language

from edsnlp.pipelines.ner.scores.base_score import SimpleScoreMatcher
from edsnlp.utils.deprecation import deprecated_factory

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

create_component = SimpleScoreMatcher
create_component = deprecated_factory(
    "score",
    "eds.score",
    assigns=["doc.ents", "doc.spans"],
)(create_component)
create_component = Language.factory(
    "eds.score",
    assigns=["doc.ents", "doc.spans"],
)(create_component)
