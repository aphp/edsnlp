from typing import Dict, List, Optional, Union

from spacy.language import Language

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
    )
