import pytest

import edsnlp
from edsnlp.utils.collections import ld_to_dl

try:
    import torch.nn
except ImportError:
    torch = None


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


@pytest.mark.parametrize("num_gpu_workers", [0, 1, 2])
@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_map_gpu(num_gpu_workers):
    import torch

    def prepare_batch(batch, device):
        return {"tensor": torch.tensor(batch).to(device)}

    def forward(batch):
        return {"outputs": batch["tensor"] * 2}

    items = range(15)
    stream = edsnlp.data.from_iterable(items)
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
    assert set(res.tolist()) == {i * 2 for i in range(15)}


# fmt: off
@pytest.mark.parametrize(
    "sort,num_cpu_workers,batch_kwargs,expected",
    [
        (False, 1, {"batch_size": 10, "batch_by": "words"}, [3, 1, 3, 1, 3, 1]),  # noqa: E501
        (False, 1, {"batch_size": 10, "batch_by": "padded_words"}, [2, 1, 1, 2, 1, 1, 2, 1, 1]),  # noqa: E501
        (False, 1, {"batch_size": 10, "batch_by": "docs"}, [10, 2]),  # noqa: E501
        (False, 2, {"batch_size": 10, "batch_by": "words"}, [2, 1, 2, 1, 2, 1, 1, 1, 1]),  # noqa: E501
        (False, 2, {"batch_size": 10, "batch_by": "padded_words"}, [2, 1, 2, 1, 2, 1, 1, 1, 1]),  # noqa: E501
        (False, 2, {"batch_size": 10, "batch_by": "docs"}, [6, 6]),  # noqa: E501
        (True, 2, {"batch_size": 10, "batch_by": "padded_words"}, [3, 3, 2, 1, 1, 1, 1]),  # noqa: E501
        (False, 2, {"batch_size": "10 words"}, [2, 1, 2, 1, 2, 1, 1, 1, 1]),  # noqa: E501
    ],
)
# fmt: on
def test_map_with_batching(sort, num_cpu_workers, batch_kwargs, expected):
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
        **batch_kwargs,
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


@pytest.mark.parametrize("shuffle_reader", [True, False])
def test_shuffle_before_generator(shuffle_reader):
    def gen_fn(x):
        yield x
        yield x

    items = [1, 2, 3, 4, 5]
    stream = edsnlp.data.from_iterable(items)
    stream = stream.map(lambda x: x)
    stream = stream.shuffle(seed=42, shuffle_reader=shuffle_reader)
    stream = stream.map(gen_fn)
    assert stream.reader.shuffle == ("dataset" if shuffle_reader else False)
    assert len(stream.ops) == (2 if shuffle_reader else 5)
    res = list(stream)
    assert res == [4, 4, 2, 2, 3, 3, 5, 5, 1, 1]


def test_shuffle_after_generator():
    def gen_fn(x):
        yield x
        yield x

    items = [1, 2, 3, 4, 5]
    stream = edsnlp.data.from_iterable(items)
    stream = stream.map(lambda x: x)
    stream = stream.map(gen_fn)
    stream = stream.shuffle(seed=43)
    assert stream.reader.shuffle == "dataset"
    assert len(stream.ops) == 5
    res = list(stream)
    assert res == [1, 2, 4, 3, 1, 3, 5, 5, 4, 2]


def test_shuffle_frozen_ml_pipeline(run_in_test_dir, frozen_ml_nlp):
    stream = edsnlp.data.read_parquet("../resources/docs.parquet", converter="omop")
    stream = stream.map_pipeline(frozen_ml_nlp, batch_size=2)
    assert len(stream.ops) == 7
    stream = stream.shuffle(batch_by="fragment")
    assert len(stream.ops) == 7
    assert stream.reader.shuffle == "fragment"


def test_unknown_shuffle():
    items = [1, 2, 3, 4, 5]
    stream = edsnlp.data.from_iterable(items)
    stream = stream.map(lambda x: x)
    with pytest.raises(ValueError):
        stream.shuffle("unknown")


def test_int_shuffle():
    items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    stream = edsnlp.data.from_iterable(items)
    stream = stream.map(lambda x: x)
    stream = stream.shuffle("2 docs", seed=42)
    assert list(stream) == [2, 1, 4, 3, 5, 6, 8, 7, 10, 9]


def test_parallel_preprocess_stop(run_in_test_dir, frozen_ml_nlp):
    nlp = frozen_ml_nlp
    stream = edsnlp.data.read_parquet(
        "../resources/docs.parquet",
        "omop",
        loop=True,
    )
    stream = stream.map(edsnlp.pipes.split(regex="\n+"))
    stream = stream.map(nlp.preprocess, kwargs=dict(supervision=True))
    stream = stream.batchify("128 words")
    stream = stream.map(nlp.collate)
    stream = stream.set_processing(num_cpu_workers=1, process_start_method="spawn")

    it = iter(stream)
    total = 0
    for _ in zip(it, range(10)):
        total += 1

    assert total == 10
    del it
