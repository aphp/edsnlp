import pytest

import edsnlp


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
