import pytest

import edsnlp
from edsnlp.utils.collections import ld_to_dl


def test_map_batches():
    items = [1, 2, 3, 4, 5]
    lazy = edsnlp.data.from_iterable(items)
    lazy = lazy.map(lambda x: x + 1)
    lazy = lazy.map_batches(lambda x: [sum(x)])
    lazy = lazy.set_processing(
        num_cpu_workers=2,
        sort_chunks=False,
        batch_size=2,
    )
    res = list(lazy)
    assert set(res) == {5, 9, 6}


@pytest.mark.parametrize("num_cpu_workers", [1, 2])
def test_flat_iterable(num_cpu_workers):
    items = [1, 2, 3, 4]
    lazy = edsnlp.data.from_iterable(items)
    lazy = lazy.set_processing(num_cpu_workers=num_cpu_workers)
    lazy = lazy.map(lambda x: [x] * x)
    res = list(lazy)
    assert sorted(res) == [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]


@pytest.mark.parametrize("num_gpu_workers", [0, 1])
def test_map_gpu(num_gpu_workers):
    import torch

    def prepare_batch(batch, device):
        return {"tensor": torch.tensor(batch).to(device)}

    def forward(batch):
        return {"outputs": batch["tensor"] * 2}

    items = [1, 2, 3, 4, 5]
    lazy = edsnlp.data.from_iterable(items)
    lazy = lazy.map(lambda x: x + 1)
    lazy = lazy.map_gpu(prepare_batch, forward)
    lazy = lazy.set_processing(
        num_gpu_workers=num_gpu_workers,
        gpu_worker_devices=["cpu"] * num_gpu_workers,
        sort_chunks=False,
        batch_size=2,
    )

    res = ld_to_dl(lazy)
    res = torch.cat(res["outputs"])
    assert set(res.tolist()) == {4, 6, 8, 10, 12}


@pytest.mark.parametrize("num_cpu_workers", [1, 2])
@pytest.mark.parametrize(
    "batch_by,expected",
    [
        ("words", [3, 1, 3, 1, 3, 1]),
        ("padded_words", [2, 1, 1, 2, 1, 1, 2, 1, 1]),
        ("docs", [10, 2]),
        ("ents", [3, 2, 3, 3, 1]),
    ],
)
def test_map_with_batching(num_cpu_workers, batch_by, expected):
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
    lazy = edsnlp.data.from_iterable(samples)
    lazy = lazy.map_pipeline(nlp)
    lazy = lazy.map_batches(len)
    lazy = lazy.to("cpu")
    lazy = lazy.set_processing(
        num_cpu_workers=num_cpu_workers,
        batch_size=10,
        batch_by=batch_by,
        chunk_size=1000,
        split_into_batches_after="matcher",
        show_progress=True,
    )
    assert list(lazy) == expected
