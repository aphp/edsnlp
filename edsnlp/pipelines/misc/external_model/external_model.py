"""`eds.dates` pipeline."""

from pathlib import Path
from typing import Any, Dict, Iterable, Union

import dill
from spacy.language import Language
from spacy.tokens import Doc

from edsnlp.pipelines.base import BaseComponent

from .wrapper.wrapper import ModelWrapper


class ExternalModel(BaseComponent):

    # noinspection PyProtectedMember
    def __init__(
        self,
        nlp: Language,
        model: Union[str, Path, ModelWrapper],
        dataset_size: int,
        extra: Dict[str, Any],
    ):
        """
        Component used to apply any wrapped model to a stream of spaCy documents.

        Parameters
        ----------
        nlp : Language
            A spaCy language
        model : Union[str, Path, ModelWrapper]
            Either a path to a wrapped model, or the model itself
        dataset_size : int
            Size (in number of examples) of the data given to the model
        extra : Dict[str, Any]
            Any additional parameters
        """

        self.nlp = nlp

        if isinstance(model, ModelWrapper):
            self.model = model
        else:
            path = Path(model)
            if path.exists():
                with open(path, "rb") as f:
                    self.model = dill.load(f)
            else:
                raise ValueError(f"Could not find model at {str(path)}")

        self.extra = extra
        self.model.prepare(**extra)
        self.dataset_size = dataset_size

    def pipe(
        self,
        stream: Iterable[Doc],
        batch_size: int,
    ):
        yield from self.model.spacy_predict(
            stream,
            dataset_size=self.dataset_size,
            batch_size=batch_size,
            **self.extra,
        )

    def __call__(self, doc: Doc):
        return next(
            self.model.spacy_predict(
                [doc],
                dataset_size=1,
                batch_size=1,
                **self.extra,
            )
        )
