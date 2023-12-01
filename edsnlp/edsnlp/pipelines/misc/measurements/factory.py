from typing import Dict, List, Optional, Union, Tuple
from typing_extensions import Literal
from spacy.language import Language
from edsnlp.pipelines.base import (
    SpanGetterArg,
    SpanSetterArg,
)
import edsnlp.pipelines.misc.measurements.patterns as patterns
from edsnlp.pipelines.misc.measurements.measurements import (
    MeasureConfig,
    MeasurementsMatcher,
    UnitConfig,
)
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    attr="NORM",
    ignore_excluded=True,
    units_config=patterns.units_config,
    number_terms=patterns.number_terms,
    value_range_terms=patterns.value_range_terms,
    unit_divisors=patterns.unit_divisors,
    measurements=None,
    stopwords_unitless=patterns.stopwords_unitless,
    stopwords_measure_unit=patterns.stopwords_measure_unit,
    measure_before_unit=False,
    parse_doc=True,
    parse_tables=True,
    all_measurements=True,
    extract_ranges=False,
    range_patterns=patterns.range_patterns,
    span_setter=None,
    span_getter=None,
    merge_mode="intersect",
    as_ents=False,
)


@Language.factory("eds.measurements", default_config=DEFAULT_CONFIG)
@deprecated_factory("eds.measures", "eds.measurements", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    measurements: Optional[Union[Dict[str, MeasureConfig], List[str]]],
    units_config: Dict[str, UnitConfig],
    number_terms: Dict[str, List[str]],
    value_range_terms: Dict[str, List[str]],
    all_measurements: bool,
    parse_tables: bool,
    parse_doc: bool,
    stopwords_unitless: List[str],
    stopwords_measure_unit: List[str],
    measure_before_unit: bool,
    unit_divisors: List[str],
    ignore_excluded: bool,
    attr: str,
    span_setter: Optional[SpanSetterArg],
    span_getter: Optional[SpanGetterArg],
    merge_mode: Literal["intersect", "align", "union"],
    extract_ranges: bool,
    range_patterns: List[Tuple[Optional[str], Optional[str]]],
    as_ents: bool,
):
    return MeasurementsMatcher(
        nlp,
        units_config=units_config,
        number_terms=number_terms,
        value_range_terms=value_range_terms,
        all_measurements=all_measurements,
        parse_tables=parse_tables,
        parse_doc=parse_doc,
        unit_divisors=unit_divisors,
        measurements=measurements,
        stopwords_unitless=stopwords_unitless,
        stopwords_measure_unit=stopwords_measure_unit,
        measure_before_unit=measure_before_unit,
        attr=attr,
        ignore_excluded=ignore_excluded,
        extract_ranges=extract_ranges,
        range_patterns=range_patterns,
        span_setter=span_setter,
        span_getter=span_getter,
        merge_mode=merge_mode,
        as_ents=as_ents,
    )
