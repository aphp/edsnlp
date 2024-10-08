from __future__ import annotations

import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from edsnlp.core.stream import Stream

doc_size_fns = {
    "words": len,
}


def execute_simple_backend(stream: Stream):
    """
    This is the default execution mode which batches the documents and processes each
    batch on the current process in a sequential manner.
    """
    try:
        torch = sys.modules["torch"]
        no_grad_ctx = torch.no_grad()
        autocast_device_type = next(
            (p.device for pipe in stream.torch_components() for p in pipe.parameters()),
            torch.device("cpu"),
        ).type.split(":")[0]
        autocast_dtype = stream.autocast if stream.autocast is not True else None
        autocast_ctx = (
            torch.autocast(
                device_type=autocast_device_type,
                dtype=autocast_dtype,
            )
            if stream.autocast
            else nullcontext()
        )
        inference_mode_ctx = (
            torch.inference_mode()
            if hasattr(torch, "inference_mode")
            else nullcontext()
        )
    except (KeyError, AttributeError):  # pragma: no cover
        no_grad_ctx = autocast_ctx = inference_mode_ctx = nullcontext()
    reader = stream.reader
    writer = stream.writer
    show_progress = stream.show_progress
    stages = stream._make_stages(split_torch_pipes=True)

    def make_torch_pipe(torch_pipe, disable_after):
        def wrapped(batches):
            for batch in batches:
                with autocast_ctx, inference_mode_ctx, no_grad_ctx:
                    batch_id = hash(tuple(id(x) for x in batch))
                    torch_pipe.enable_cache(batch_id)
                    batch = torch_pipe.batch_process(batch)
                    if disable_after:
                        torch_pipe.disable_cache(batch_id)
                yield batch

        return wrapped

    def process():
        bar = tqdm(smoothing=0.1, mininterval=5.0, disable=not show_progress)

        with bar, stream.eval():
            items = reader.read_records()

            for stage_idx, stage in enumerate(stages):
                for op in stage.cpu_ops:
                    items = op(items)

                if stage.gpu_op is not None:
                    pipe = make_torch_pipe(stage.gpu_op, stage_idx == len(stages) - 2)
                    items = pipe(items)

            if writer is not None:
                items = (writer.handle_record(item) for item in items)

            if getattr(writer, "batch_by", None) is not None:
                items = writer.batch_by(items, writer.batch_size)
                # get the 1st element (2nd is the count)
                for b in items:
                    item, count = writer.handle_batch(b)
                    bar.update(count)
                    yield item
            else:
                for item in items:
                    bar.update(1)
                    yield item

    items = process()

    if writer is not None:
        items = writer.consolidate(items)

    return items
