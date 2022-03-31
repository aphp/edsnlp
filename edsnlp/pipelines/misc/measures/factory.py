from typing import Dict, List, Union

from spacy.language import Language

from edsnlp.pipelines.misc.measures import Measures

DEFAULT_CONFIG = dict(
    attr="NORM",
    ignore_excluded=False,
    measures=["eds.measures.size", "eds.measures.weight", "eds.measures.angle"],
)


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
