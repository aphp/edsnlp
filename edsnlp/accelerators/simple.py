from contextlib import nullcontext
from typing import Any, Dict, Iterable

from edsnlp.core.registry import registry
from edsnlp.utils.collections import batchify

from .base import Accelerator, FromDictFieldsToDoc, FromDoc, ToDoc


@registry.accelerator.register("simple")
class SimpleAccelerator(Accelerator):
    """
    This is the simplest accelerator which batches the documents and process each batch
    on the main process (the one calling `.pipe()`).

    Examples
    --------

    ```{ .python .no-check }
    docs = list(nlp.pipe([content1, content2, ...]))
    ```

    or, if you want to override the model defined batch size

    ```{ .python .no-check }
    docs = list(nlp.pipe([content1, content2, ...], batch_size=8))
    ```

    which is equivalent to passing a confit dict

    ```{ .python .no-check }
    docs = list(
        nlp.pipe(
            [text1, text2, ...],
            accelerator={
                "@accelerator": "simple",
                "batch_size": 8,
            },
        )
    )
    ```

    or the instantiated accelerator directly

    ```{ .python .no-check }
    from edsnlp.accelerators.simple import SimpleAccelerator

    accelerator = SimpleAccelerator(batch_size=8)
    docs = list(nlp.pipe([content1, content2, ...], accelerator=accelerator))
    ```

    If you have a GPU, make sure to move the model to the appropriate device before
    calling `.pipe()`. If you have multiple GPUs, use the
    [multiprocessing][edsnlp.accelerators.multiprocessing.MultiprocessingAccelerator]
    accelerator instead.

    ```{ .python .no-check }
    nlp.to("cuda")
    docs = list(nlp.pipe([content1, content2, ...]))
    ```

    Parameters
    ----------
    batch_size: int
        The number of documents to process in each batch.
    """

    def __init__(
        self,
        *,
        batch_size: int = 32,
    ):
        self.batch_size = batch_size

    def __call__(
        self,
        inputs: Iterable[Any],
        nlp: Any,
        to_doc: ToDoc = FromDictFieldsToDoc("content"),
        from_doc: FromDoc = lambda doc: doc,
        component_cfg: Dict[str, Dict[str, Any]] = None,
    ):
        try:
            import torch

            no_grad = torch.no_grad
        except ImportError:
            no_grad = nullcontext
        docs = (to_doc(doc, nlp) for doc in inputs)
        for batch in batchify(docs, batch_size=self.batch_size):
            with no_grad(), nlp.cache(), nlp.train(False):
                for name, pipe in nlp.pipeline:
                    if name not in nlp._disabled:
                        if hasattr(pipe, "batch_process"):
                            batch = pipe.batch_process(batch)
                        else:
                            batch = [pipe(doc) for doc in batch]  # type: ignore
            yield from (from_doc(doc) for doc in batch)
