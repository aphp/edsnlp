from __future__ import annotations

import sys
from contextlib import nullcontext

from tqdm import tqdm

from edsnlp.core.stream import (
    Stream,
    StreamRecord,
    StreamSentinel,
    unwrap_stream_record,
    wrap_stream_record,
    wrap_stream_record_batch,
)
from edsnlp.utils.batching import BatchTimeoutSentinel

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
        device = next(
            (p.device for pipe in stream.torch_components() for p in pipe.parameters()),
            torch.device("cpu"),
        )
        device_type = getattr(device, "type", device).split(":")[0]
        autocast = stream.autocast
        autocast_ctx = nullcontext()
        try:
            if autocast:
                autocast_ctx = torch.autocast(
                    device_type=device_type,
                    dtype=autocast if autocast is not True else None,
                )
        except RuntimeError:  # pragma: no cover
            pass

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
                if isinstance(batch, BatchTimeoutSentinel):  # pragma: no cover
                    yield batch
                    continue
                if isinstance(batch, StreamSentinel):  # pragma: no cover
                    yield batch
                    continue
                if batch and isinstance(batch[0], StreamRecord):
                    pending = [item for item in batch if item.error is None]
                    if not pending:  # pragma: no cover
                        yield batch
                        continue
                else:
                    pending = batch
                with autocast_ctx, inference_mode_ctx, no_grad_ctx:
                    batch_id = hash(tuple(id(x) for x in pending))
                    torch_pipe.enable_cache(batch_id)
                    try:
                        res = torch_pipe.batch_process(
                            [unwrap_stream_record(item) for item in pending]
                        )
                    except BaseException as exc:  # noqa: BLE001
                        if batch and isinstance(batch[0], StreamRecord):
                            yield [
                                item
                                if item.error is not None
                                else wrap_stream_record(item, item.value, error=exc)
                                for item in batch
                            ]
                            continue
                        raise  # pragma: no cover
                    finally:
                        if disable_after:
                            torch_pipe.disable_cache(batch_id)
                    if pending is batch:
                        batch = wrap_stream_record_batch(batch, res)
                    else:  # pragma: no cover
                        processed = iter(wrap_stream_record_batch(pending, res))
                        batch = [
                            item if item.error is not None else next(processed)
                            for item in batch
                        ]
                yield batch

        return wrapped

    def process():
        bar = tqdm(smoothing=0.1, mininterval=5.0, disable=not show_progress)

        with bar, stream.eval():
            items = reader.read_records()
            items = (
                task
                for item in items
                for task in (
                    (item,)
                    if isinstance(item, StreamSentinel)
                    or is_batch_timeout_sentinel(item)
                    else reader.extract_task(item)
                )
            )

            for stage_idx, stage in enumerate(stages):
                for op in stage.cpu_ops:
                    items = op(items)

                if stage.gpu_op is not None:
                    pipe = make_torch_pipe(stage.gpu_op, stage_idx == len(stages) - 2)
                    items = pipe(items)

            if writer is not None:
                items = (
                    item
                    if isinstance(item, StreamSentinel)
                    or is_batch_timeout_sentinel(item)
                    else writer.handle_record(item)
                    for item in items
                )

            if getattr(writer, "batch_fn", None) is not None:
                items = writer.batch_fn(items, writer.batch_size, sentinel_mode="drop")
                # get the 1st element (2nd is the count)
                for b in items:
                    if isinstance(b, StreamSentinel) or is_batch_timeout_sentinel(b):
                        continue
                    item, count = writer.handle_batch(b)
                    bar.update(count)
                    yield item
            else:
                for item in items:
                    if is_batch_timeout_sentinel(item):
                        continue
                    if not isinstance(item, StreamSentinel):
                        bar.update(1)
                        yield item

    items = process()

    if writer is not None:
        items = writer.consolidate(items)

    return items
