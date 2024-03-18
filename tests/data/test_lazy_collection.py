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
