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
    unit_divisors=patterns.unit_divisors,
    measurements=None,
    stopwords=patterns.stopwords,
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
    attr: str,
):
    return MeasurementsMatcher(
        nlp,
        units_config=units_config,
        number_terms=number_terms,
        unit_divisors=unit_divisors,
        measurements=measurements,
        stopwords=stopwords,
        attr=attr,
        ignore_excluded=ignore_excluded,
    )
