from typing import Dict, List, Optional, Tuple, Union

from spacy.language import Language

import edsnlp.pipelines.misc.measurements.patterns as patterns
from edsnlp.pipelines.misc.measurements.measurements import (
    MeasureConfig,
    MeasurementsMatcher,
    MergeStrategy,
    UnitConfig,
)
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    attr="NORM",
    measurements=None,
    ignore_excluded=True,
    units_config=patterns.units_config,
    number_terms=patterns.number_terms,
    unit_divisors=patterns.unit_divisors,
    stopwords=patterns.stopwords,
    compose_units=True,
    extract_ranges=False,
    range_patterns=patterns.range_patterns,
    as_ents=False,
    merge_mode=MergeStrategy.union,
)


@Language.factory("eds.measurements", default_config=DEFAULT_CONFIG)
@deprecated_factory("eds.measures", "eds.measurements", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    measurements: Optional[Union[Dict[str, MeasureConfig], List[str]]],
    units_config: Dict[str, UnitConfig],
    number_terms: Dict[str, List[str]],
    stopwords: List[str],
    unit_divisors: List[str],
    ignore_excluded: bool,
    compose_units: bool,
    attr: str,
    extract_ranges: bool,
    range_patterns: List[Tuple[Optional[str], Optional[str]]],
    as_ents: bool,
    merge_mode: MergeStrategy,
):
    return MeasurementsMatcher(
        nlp,
        measurements=measurements,
        units_config=units_config,
        number_terms=number_terms,
        stopwords=stopwords,
        unit_divisors=unit_divisors,
        ignore_excluded=ignore_excluded,
        compose_units=compose_units,
        attr=attr,
        extract_ranges=extract_ranges,
        range_patterns=range_patterns,
        as_ents=as_ents,
        merge_mode=merge_mode,
    )
