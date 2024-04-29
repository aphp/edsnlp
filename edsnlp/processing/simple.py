from __future__ import annotations

import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING

from edsnlp.utils.batching import batchify_fns
from edsnlp.utils.collections import batchify, flatten

from .utils import apply_basic_pipes

if TYPE_CHECKING:
    from edsnlp.core.lazy_collection import LazyCollection

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
    if split_into_batches_after is None and (lc.batch_by != "docs" or lc.sort_chunks):
        split_into_batches_after = next(
            (s[0] for s in lc.pipeline if s[0] == "_ensure_doc"), None
        )
    names = [None] + [step[0] for step in lc.pipeline]
    chunk_components = lc.pipeline[: names.index(split_into_batches_after)]
    batch_components = lc.pipeline[names.index(split_into_batches_after) :]

    def process():
        bar = nullcontext()
        if show_progress:
            from tqdm import tqdm

            bar = tqdm(smoothing=0.1, mininterval=5.0)

        with bar, lc.eval():
            for docs in batchify(
                (
                    subtask
                    for task, count in reader.read_main()
                    for subtask in reader.read_worker([task])
                ),
                batch_size=lc.chunk_size,
            ):
                docs = apply_basic_pipes(docs, chunk_components)

                if lc.sort_chunks:
                    docs.sort(key=doc_size_fns.get(lc.sort_chunks, len))

                for batch in batchify_fns[lc.batch_by](docs, lc.batch_size):
                    count = len(batch)
                    with no_grad(), lc.cache():
                        batch = apply_basic_pipes(batch, batch_components)

                    if writer is not None:
                        result, count = writer.write_worker(batch)
                        if show_progress:
                            bar.update(count)
                        yield result
                    else:
                        if show_progress:
                            bar.update(count)
                        yield batch
            if writer is not None:
                result, count = writer.finalize()
                if show_progress:
                    bar.update(count)
                if count:
                    yield result

    gen = process()
    return flatten(gen) if writer is None else writer.write_main(gen)
