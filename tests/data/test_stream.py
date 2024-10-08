import pytest

import edsnlp
from edsnlp.utils.collections import ld_to_dl


def test_map_batches():
    items = [1, 2, 3, 4, 5]
    stream = edsnlp.data.from_iterable(items)
    stream = stream.map(lambda x: x + 1)  # 2, 3, 4, 5, 6
    stream = stream.map_batches(lambda x: [sum(x)])
    stream = stream.set_processing(
        num_cpu_workers=2,
        sort_chunks=False,
        batch_size=2,
    )
    res = list(stream)
    assert res == [6, 8, 6]  # 2+4, 3+5, 6


@pytest.mark.parametrize("num_cpu_workers", [1, 2])
def test_flat_iterable(num_cpu_workers):
    items = [1, 2, 3, 4]
    stream = edsnlp.data.from_iterable(items)
    stream = stream.set_processing(num_cpu_workers=num_cpu_workers)
    stream = stream.map(lambda x: [x] * x)
    stream = stream.flatten()
    res = list(stream.to_iterable(converter=lambda x: x))
    assert sorted(res) == [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]


@pytest.mark.parametrize("num_gpu_workers", [0, 1])
def test_map_gpu(num_gpu_workers):
    import torch

    def prepare_batch(batch, device):
        return {"tensor": torch.tensor(batch).to(device)}

    def forward(batch):
        return {"outputs": batch["tensor"] * 2}

    items = [1, 2, 3, 4, 5]
    stream = edsnlp.data.from_iterable(items)
    stream = stream.map(lambda x: x + 1)
    if num_gpu_workers == 0:
        # this is just to fuse tests, and test map_gpu
        # following a map_batches without specifying a batch size
        stream = stream.map_batches(lambda x: x)
    stream = stream.map_gpu(prepare_batch, forward)
    stream = stream.set_processing(
        num_gpu_workers=num_gpu_workers,
        gpu_worker_devices=["cpu"] * num_gpu_workers,
        sort_chunks=False,
        batch_size=2,
    )

    res = ld_to_dl(stream)
    res = torch.cat(res["outputs"])
    assert set(res.tolist()) == {4, 6, 8, 10, 12}


@pytest.mark.parametrize(
    "sort,num_cpu_workers,batch_by,expected",
    [
        (False, 1, "words", [3, 1, 3, 1, 3, 1]),
        (False, 1, "padded_words", [2, 1, 1, 2, 1, 1, 2, 1, 1]),
        (False, 1, "docs", [10, 2]),
        (False, 2, "words", [2, 1, 2, 1, 2, 1, 1, 1, 1]),
        (False, 2, "padded_words", [2, 1, 2, 1, 2, 1, 1, 1, 1]),
        (False, 2, "docs", [6, 6]),
        (True, 2, "padded_words", [3, 3, 2, 1, 1, 1, 1]),
    ],
)
def test_map_with_batching(sort, num_cpu_workers, batch_by, expected):
    nlp = edsnlp.blank("eds")
    nlp.add_pipe(
        "eds.matcher",
        config={
            "terms": {
                "foo": ["This", "is", "a", "sentence", ".", "Short", "snippet", "too"],
            }
        },
        name="matcher",
    )
    samples = [
        "This is a sentence.",
        "Short snippet",
        "Short snippet too",
        "This is a very very long sentence that will make more than 10 words",
    ] * 3
    stream = edsnlp.data.from_iterable(samples)
    if sort:
        stream = stream.map_batches(lambda x: sorted(x, key=len), batch_size=1000)
    stream = stream.map_pipeline(nlp)
    stream = stream.map_batches(len)
    stream = stream.set_processing(
        num_cpu_workers=num_cpu_workers,
        batch_size=10,
        batch_by=batch_by,
        chunk_size=1000,  # deprecated
        split_into_batches_after="matcher",
        show_progress=True,
    )
    assert list(stream) == expected


def test_repr(frozen_ml_nlp, tmp_path):
    items = ["ceci est un test", "ceci est un autre test"]
    stream = (
        edsnlp.data.from_iterable(items, converter=frozen_ml_nlp.make_doc)
        .map(lambda x: x)
        .map_pipeline(frozen_ml_nlp, batch_size=2)
        .map_batches(lambda b: sorted(b, key=len))
        .set_processing(num_cpu_workers=2)
        .write_json(tmp_path / "out_test.jsonl", lines=True, execute=False)
    )
    assert "Stream" in repr(stream)
