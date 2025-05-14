# ruff: noqa: E501
"""
This module is meant to be run as a script to test multiprocessing
with a pipe declared in the main module.
Without the dill settings in multiprocessing.py
```
    dill.settings["recurse"] = False
    dill.settings["byref"] = True
```
this fails with an error like
```
_pickle.PicklingError: Can't pickle <cyfunction ValidatedFunction.create_model.<locals>.DecoratorBaseModel.check_args at 0x...>:
it's not found as pydantic.decorator.ValidatedFunction.create_model.<locals>.DecoratorBaseModel.check_args
```
"""

from typing import Optional

from spacy.tokens import Doc

import edsnlp
from edsnlp import registry
from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import BaseComponent


class SimplePipe(BaseComponent):
    def __init__(
        self,
        nlp: PipelineProtocol,
        name: Optional[str] = "simple_pipe",
    ):
        super().__init__(nlp=nlp, name=name)

    def __call__(self, doc: Doc) -> Doc:
        return doc


if __name__ == "__main__":
    create_component = registry.factory.register(
        "test_simple_pipe",
        assigns=["doc.spans"],
    )(SimplePipe)

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(create_component())

    docs = [f"Texte {i}" for i in range(100)]
    docs = nlp.pipe(docs)
    docs = docs.set_processing(backend="multiprocessing")
    list(docs)
