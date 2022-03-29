from typing import Dict, List, Union

from spacy.language import Language

from edsnlp.pipelines.misc.measures import Measures
from edsnlp.utils.deprecation import deprecated_factory

DEFAULT_CONFIG = dict(
    attr="NORM",
    ignore_excluded=False,
    measures=["eds.measures.size", "eds.measures.weight", "eds.measures.angle"],
)


@deprecated_factory("measures", "eds.measures", default_config=DEFAULT_CONFIG)
@Language.factory("eds.measures", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    measures: Union[str, List[str], Dict[str, Dict]],
    attr: str,
    ignore_excluded: bool,
):
    return Measures(
        nlp,
        measures=measures,
        attr=attr,
        ignore_excluded=ignore_excluded,
    )
