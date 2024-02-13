from __future__ import annotations

import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING

from edsnlp.data.converters import set_current_tokenizer
from edsnlp.utils.collections import batchify, flatten_once

if TYPE_CHECKING:
    from edsnlp.core.lazy_collection import LazyCollection


batch_size_fns = {
    "words": lambda batch: sum(len(doc) for doc in batch),
    "padded_words": lambda batch: max(len(doc) for doc in batch) * len(batch),
    "docs": len,
}

doc_size_fns = {
    "words": len,
}


def execute_simple_backend(
    lc: LazyCollection,
):
    """
    This is the default execution mode which batches the documents and processes each
    batch on the current process in a sequential manner.
    """
    try:
        no_grad = sys.modules["torch"].no_grad
    except (KeyError, AttributeError):
        no_grad = nullcontext
    reader = lc.reader
    writer = lc.writer
    show_progress = lc.show_progress

    split_into_batches_after = lc.split_into_batches_after
    if split_into_batches_after is None or lc.batch_by != "docs" or lc.sort_chunks:
        split_into_batches_after = next(
            (p[0] for p in lc.pipeline if p[0] is not None), None
        )
    names = [step[0] for step in lc.pipeline] + [None]
    chunk_components = lc.pipeline[: names.index(split_into_batches_after)]
    batch_components = lc.pipeline[names.index(split_into_batches_after) :]

    def process():
        bar = nullcontext()
        if show_progress:
            from tqdm import tqdm

            bar = tqdm(smoothing=0.1, mininterval=5.0)

        with bar:
            for docs in batchify(
                (
                    subtask
                    for task, count in reader.read_main()
                    for subtask in reader.read_worker([task])
                ),
                batch_size=lc.chunk_size,
            ):
                for name, pipe, kwargs, tokenizer in chunk_components:
                    with set_current_tokenizer(tokenizer):
                        if hasattr(pipe, "batch_process"):
                            docs = pipe.batch_process(docs, **kwargs)
                        else:
                            docs = [pipe(doc, **kwargs) for doc in docs]  #

                if lc.sort_chunks:
                    docs.sort(key=doc_size_fns.get(lc.sort_chunks, len))

                batches = [
                    batch
                    for batch in batchify(
                        docs,
                        batch_size=lc.batch_size,
                        formula=doc_size_fns.get(lc.batch_by, len),
                    )
                ]

                for batch in batches:
                    with no_grad(), lc.cache():
                        for name, pipe, kwargs, tokenizer in batch_components:
                            with set_current_tokenizer(tokenizer):
                                if hasattr(pipe, "batch_process"):
                                    batch = pipe.batch_process(batch, **kwargs)
                                else:
                                    batch = [pipe(doc, **kwargs) for doc in batch]  # type: ignore

                    if writer is not None:
                        result, count = writer.write_worker(batch)
                        if show_progress:
                            bar.update(count)
                        yield result
                    else:
                        if show_progress:
                            bar.update(len(batch))
                        yield batch
            if writer is not None:
                result, count = writer.finalize()
                if show_progress:
                    bar.update(count)
                if count:
                    yield result

    gen = process()
    return flatten_once(gen) if writer is None else writer.write_main(gen)
