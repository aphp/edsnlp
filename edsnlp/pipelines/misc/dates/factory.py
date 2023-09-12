from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from .dates import DatesMatcher

DEFAULT_CONFIG = dict(
    absolute=None,
    relative=None,
    duration=None,
    false_positive=None,
    on_ents_only=False,
    span_getter=None,
    merge_mode="intersect",
    detect_periods=False,
    detect_time=True,
    period_proximity_threshold=3,
    as_ents=False,
    attr="LOWER",
    date_label="date",
    duration_label="duration",
    period_label="period",
    span_setter={
        "dates": ["date"],
        "durations": ["duration"],
        "periods": ["period"],
    },
)

create_component = deprecated_factory(
    "dates",
    "eds.dates",
    assigns=["doc.spans", "doc.ents"],
)(DatesMatcher)
create_component = Language.factory(
    "eds.dates",
    assigns=["doc.spans", "doc.ents"],
)(create_component)
