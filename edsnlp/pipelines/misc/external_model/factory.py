from pathlib import Path
from typing import Any, Dict, Union

from spacy.language import Language

from . import ExternalModel
from .wrapper.wrapper import ModelWrapper

DEFAULT_CONFIG = dict(
    dataset_size=1000,
    extra={},
)


@Language.factory("eds.external-model", default_config=DEFAULT_CONFIG)
def create_component(
    nlp: Language,
    name: str,
    model: Union[ModelWrapper, Path],
    dataset_size: int,
    extra: Dict[str, Any],
):
    return ExternalModel(
        nlp,
        model=model,
        dataset_size=dataset_size,
        extra=extra,
    )
