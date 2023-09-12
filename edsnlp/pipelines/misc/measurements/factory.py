from spacy.language import Language

from edsnlp.utils.deprecation import deprecated_factory

from . import patterns
from .measurements import MeasurementsMatcher

DEFAULT_CONFIG = dict(
    measurements=list(patterns.common_measurements.keys()),  # noqa: E501
    units_config=patterns.units_config,
    number_terms=patterns.number_terms,
    stopwords=patterns.stopwords,
    unit_divisors=patterns.unit_divisors,
    ignore_excluded=True,
    compose_units=True,
    attr="NORM",
    extract_ranges=False,
    range_patterns=patterns.range_patterns,
    after_snippet_limit=6,
    before_snippet_limit=10,
    span_getter=None,
    merge_mode="intersect",
    as_ents=False,
    span_setter=None,
)

create_component = deprecated_factory(
    "eds.measures",
    "eds.measurements",
    assigns=["doc.spans", "doc.ents"],
)(MeasurementsMatcher)
create_component = Language.factory(
    "eds.measurements",
    assigns=["doc.spans", "doc.ents"],
)(create_component)
