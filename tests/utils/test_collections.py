from edsnlp.utils.collections import (
    batch_compress_dict,
    batchify,
    decompress_dict,
    dl_to_ld,
    flatten,
    flatten_once,
    get_deep_attr,
    ld_to_dl,
    multi_tee,
    set_deep_attr,
)


def test_multi_tee():
    gen = (i for i in range(10))
    tee = multi_tee(gen)
    items1 = [value for i, value in zip(tee, range(5))]
    items2 = [value for i, value in zip(tee, range(5))]
    assert items1 == items2 == [0, 1, 2, 3, 4]

    # not the behavior I'd like (continue from 5 would be nice) but at least
    # the generator is not exhausted
    assert next(gen) == 6

    assert multi_tee(items1) is items1


def test_flatten():
    items = [1, [2, 3], [[4, 5], 6], [[[7, 8], 9], 10]]
    assert list(flatten_once(items)) == [1, 2, 3, [4, 5], 6, [[7, 8], 9], 10]
    assert list(flatten(items)) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_dict_of_lists():
    items = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert ld_to_dl(items) == {"a": [1, 3], "b": [2, 4]}
    assert list(dl_to_ld({"a": [1, 3], "b": [2, 4]})) == items


def test_dict_compression():
    list1 = [1, 2, 3]
    mapping1 = {"a": list1, "b": {"c": list1, "d": 4}}
    list2 = [1, 2, 3]
    mapping2 = {"a": list2, "b": {"c": list2, "d": 4}}

    samples = [mapping1, mapping2]
    assert list(batch_compress_dict(samples)) == [
        {"a|b/c": [1, 2, 3], "b/d": 4},
        {"a|b/c": [1, 2, 3], "b/d": 4},
    ]

    res = decompress_dict(
        [
            {"a|b/c": [1, 2, 3], "b/d": 4},
            {"a|b/c": [1, 2, 3], "b/d": 4},
        ]
    )
    assert res == {
        "a": [[1, 2, 3], [1, 2, 3]],
        "b": {"c": [[1, 2, 3], [1, 2, 3]], "d": [4, 4]},
    }
    assert res["a"] is res["b"]["c"]


def test_batchify():
    items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batches = list(batchify(items, 3))
    assert len(batches) == 4
    batches = list(batches)
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_deep_path():
    class custom:
        def __init__(self, values):
            self.values = values

    item = {"a": {"b": (0, 1), "other": custom((1, 2))}}
    assert get_deep_attr(item, "a.b.0") == 0
    assert get_deep_attr(item, "a.other.values.0") == 1
    set_deep_attr(item, "a.b.1", 2)
    set_deep_attr(item, "a.other.values.0", 1000)
    assert item["a"]["b"] == (0, 2)
    assert item["a"]["other"].values == (1000, 2)
