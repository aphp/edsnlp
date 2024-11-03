import pytest

from edsnlp.utils.batching import (
    DATASET_END_SENTINEL,
    BatchSizeArg,
    FragmentEndSentinel,
    StreamSentinel,
    batchify,
    batchify_by_dataset,
    batchify_by_fragment,
    batchify_by_length_sum,
    batchify_by_padded,
    stat_batchify,
)


class MockStreamSentinel(StreamSentinel):
    pass


# Tests for BatchSizeArg
def test_batch_size_arg_validate():
    # Valid inputs
    assert BatchSizeArg.validate("10 samples") == (10, "samples")
    assert BatchSizeArg.validate("20 words") == (20, "words")
    assert BatchSizeArg.validate(15) == (15, "docs")
    assert BatchSizeArg.validate("docs") == (None, "docs")
    assert BatchSizeArg.validate("tokens") == (None, "tokens")
    assert BatchSizeArg.validate("25") == (25, "docs")

    # Invalid inputs
    with pytest.raises(Exception):
        BatchSizeArg.validate("invalid input")
    with pytest.raises(Exception):
        BatchSizeArg.validate("10 invalid input")
    with pytest.raises(Exception):
        BatchSizeArg.validate("invalid input 10")


# Tests for batchify function
def test_batchify_simple():
    data = [1, 2, 3, 4, 5]
    batches = list(batchify(data, batch_size=2))
    assert batches == [[1, 2], [3, 4], [5]]


def test_batchify_drop_last():
    data = [1, 2, 3, 4, 5]
    batches = list(batchify(data, batch_size=2, drop_last=True))
    assert batches == [[1, 2], [3, 4]]


def test_batchify_sentinel_drop():
    data = [1, 2, MockStreamSentinel(), 3, 4]
    batches = list(batchify(data, batch_size=2, sentinel_mode="drop"))
    assert batches == [[1, 2], [3, 4]]


def test_batchify_sentinel_keep():
    sentinel = MockStreamSentinel()
    data = [1, 2, sentinel, 3, 4]
    batches = list(batchify(data, batch_size=2, sentinel_mode="keep"))
    assert batches == [[1, 2, sentinel], [3, 4]]


def test_batchify_sentinel_split():
    sentinel = MockStreamSentinel()
    data = [1, 2, sentinel, 3, 4]
    batches = list(batchify(data, batch_size=2, sentinel_mode="split"))
    assert batches == [[1, 2], sentinel, [3, 4]]


# Tests for batchify_by_length_sum
def test_batchify_by_length_sum_simple():
    data = ["a", "bb", "ccc", "dddd", "eeeee"]
    batches = list(batchify_by_length_sum(data, batch_size=5))
    assert batches == [["a", "bb"], ["ccc"], ["dddd"], ["eeeee"]]


def test_batchify_by_length_sum_drop_last():
    data = ["a", "bb", "ccc", "dddd", "eeeee"]
    batches = list(batchify_by_length_sum(data, batch_size=5, drop_last=True))
    assert batches == [["a", "bb"], ["ccc"], ["dddd"]]


# Tests for batchify_by_length_sum
def test_batchify_by_length_sum_split():
    sentinel = MockStreamSentinel()
    data = ["aa", "bb", sentinel, "ccc", "dddd", "eeeee"]
    batches = list(batchify_by_length_sum(data, batch_size=7, sentinel_mode="split"))
    assert batches == [["aa", "bb"], sentinel, ["ccc", "dddd"], ["eeeee"]]


# Tests for batchify_by_length_sum
def test_batchify_by_length_sum_keep():
    sentinel = MockStreamSentinel()
    data = ["aa", "bb", sentinel, "ccc", "dddd", "eeeee"]
    batches = list(batchify_by_length_sum(data, batch_size=7, sentinel_mode="keep"))
    assert batches == [["aa", "bb", sentinel, "ccc"], ["dddd"], ["eeeee"]]


# Tests for batchify_by_padded
def test_batchify_by_padded_simple():
    data = ["a", "bb", "ccc", "dddd"]
    batches = list(batchify_by_padded(data, batch_size=6))
    assert batches == [["a", "bb"], ["ccc"], ["dddd"]]


def test_batchify_by_padded_drop_last():
    data = ["a", "bb", "ccc", "dddd"]
    batches = list(batchify_by_padded(data, batch_size=6, drop_last=True))
    assert batches == [["a", "bb"], ["ccc"]]


def test_batchify_by_padded_sentinel_keep():
    sentinel = MockStreamSentinel()
    data = ["a", sentinel, "bb", "ccc"]
    batches = list(batchify_by_padded(data, batch_size=6, sentinel_mode="keep"))
    assert batches == [["a", sentinel, "bb"], ["ccc"]]


def test_batchify_by_padded_sentinel_split():
    sentinel = MockStreamSentinel()
    data = ["a", sentinel, "bb", "ccc"]
    batches = list(batchify_by_padded(data, batch_size=5, sentinel_mode="split"))
    assert batches == [["a"], sentinel, ["bb"], ["ccc"]]


# Tests for batchify_by_dataset
def test_batchify_by_dataset_simple():
    data = [
        "item1",
        "item2",
        DATASET_END_SENTINEL,
        "item3",
        DATASET_END_SENTINEL,
        "item4",
        "item5",
    ]
    batches = list(batchify_by_dataset(data))
    assert batches == [
        ["item1", "item2"],
        DATASET_END_SENTINEL,
        ["item3"],
        DATASET_END_SENTINEL,
        ["item4", "item5"],
    ]


def test_batchify_by_dataset_sentinel_split():
    sentinel = MockStreamSentinel()
    data = ["item1", sentinel, "item2", DATASET_END_SENTINEL, "item3"]
    batches = list(batchify_by_dataset(data, sentinel_mode="split"))
    assert batches == [["item1"], sentinel, ["item2"], DATASET_END_SENTINEL, ["item3"]]


def test_batchify_by_dataset_sentinel_keep():
    sentinel = MockStreamSentinel()
    data = ["item1", sentinel, "item2", DATASET_END_SENTINEL, "item3"]
    batches = list(batchify_by_dataset(data, sentinel_mode="keep"))
    assert batches == [["item1", sentinel, "item2"], DATASET_END_SENTINEL, ["item3"]]


def test_batchify_by_dataset_sentinel_drop():
    sentinel = MockStreamSentinel()
    data = ["item1", sentinel, "item2", DATASET_END_SENTINEL, "item3"]
    batches = list(batchify_by_dataset(data, sentinel_mode="drop"))
    assert batches == [["item1", "item2"], DATASET_END_SENTINEL, ["item3"]]


def test_batchify_by_dataset_drop_last():
    data = ["item1", "item2", DATASET_END_SENTINEL, "item3"]
    batches = list(batchify_by_dataset(data, drop_last=True))
    assert batches == [["item1", "item2"], DATASET_END_SENTINEL]


# Tests for batchify_by_fragment
def test_batchify_by_fragment_simple():
    fragment_end_1 = FragmentEndSentinel("fragment1")
    fragment_end_2 = FragmentEndSentinel("fragment2")
    data = ["item1", "item2", fragment_end_1, "item3", fragment_end_2, "item4"]
    batches = list(batchify_by_fragment(data))
    assert batches == [
        ["item1", "item2"],
        fragment_end_1,
        ["item3"],
        fragment_end_2,
        ["item4"],
    ]


def test_batchify_by_fragment_sentinel_split():
    sentinel = MockStreamSentinel()
    fragment_end = FragmentEndSentinel("fragment")
    data = ["item1", sentinel, "item2", fragment_end]
    batches = list(batchify_by_fragment(data, sentinel_mode="split"))
    assert batches == [["item1"], sentinel, ["item2"], fragment_end]


def test_batchify_by_fragment_sentinel_keep():
    sentinel = MockStreamSentinel()
    fragment_end = FragmentEndSentinel("fragment")
    data = ["item1", sentinel, "item2", fragment_end]
    batches = list(batchify_by_fragment(data, sentinel_mode="keep"))
    assert batches == [["item1", sentinel, "item2"], fragment_end]


def test_batchify_by_fragment_sentinel_drop():
    sentinel = MockStreamSentinel()
    fragment_end = FragmentEndSentinel("fragment")
    data = ["item1", sentinel, "item2", fragment_end]
    batches = list(batchify_by_fragment(data, sentinel_mode="drop"))
    assert batches == [["item1", "item2"], fragment_end]


def test_batchify_by_fragment_drop_last():
    fragment_end = FragmentEndSentinel("fragment")
    data = ["item1", "item2", fragment_end]
    batches = list(batchify_by_fragment(data, sentinel_mode="split", drop_last=True))
    assert batches == [["item1", "item2"], fragment_end]


# Tests for stat_batchify
def test_stat_batchify_simple():
    data = [
        {"/stats/length": 2, "text": "aa"},
        {"/stats/length": 3, "text": "bbb"},
        {"/stats/length": 4, "text": "cccc"},
        {"/stats/length": 2, "text": "dd"},
    ]
    batch_fn = stat_batchify("length")
    batches = list(batch_fn(data, batch_size=5))
    assert batches == [
        [data[0], data[1]],  # Total length: 5
        [data[2]],  # Total length: 4
        [data[3]],  # Total length: 2
    ]


def test_stat_batchify_invalid_key():
    data = [{"text": "aaa"}]
    batch_fn = stat_batchify("length")
    with pytest.raises(ValueError):
        list(batch_fn(data, batch_size=5))


def test_stat_batchify_sentinel_split():
    sentinel = MockStreamSentinel()
    data = [
        {"/stats/length": 2, "text": "aa"},
        sentinel,
        {"/stats/length": 3, "text": "bbb"},
    ]
    batch_fn = stat_batchify("length")
    batches = list(batch_fn(data, batch_size=5, sentinel_mode="split"))
    assert batches == [
        [data[0]],
        sentinel,
        [data[2]],
    ]


def test_stat_batchify_sentinel_keep():
    sentinel = MockStreamSentinel()
    data = [
        {"/stats/length": 2, "text": "aa"},
        sentinel,
        {"/stats/length": 4, "text": "bbbb"},
    ]
    batch_fn = stat_batchify("length")
    batches = list(batch_fn(data, batch_size=5, sentinel_mode="keep"))
    assert batches == [
        [data[0], sentinel],
        [data[2]],
    ]


def test_stat_batchify_drop_last():
    data = [
        {"/stats/length": 2, "text": "aa"},
        {"/stats/length": 3, "text": "bbb"},
        {"/stats/length": 4, "text": "cccc"},
    ]
    batch_fn = stat_batchify("length")
    batches = list(batch_fn(data, batch_size=6, drop_last=True))
    assert batches == [
        [data[0], data[1]],  # Total length: 5
    ]  # Last batch is dropped because total length is 4


# Additional tests to ensure full coverage
def test_batchify_empty_iterable():
    data = []
    batches = list(batchify(data, batch_size=2))
    assert batches == []


def test_batchify_by_length_sum_empty_iterable():
    data = []
    batches = list(batchify_by_length_sum(data, batch_size=5))
    assert batches == []


def test_batchify_by_padded_empty_iterable():
    data = []
    batches = list(batchify_by_padded(data, batch_size=6))
    assert batches == []


def test_batchify_by_dataset_empty_iterable():
    data = []
    batches = list(batchify_by_dataset(data))
    assert batches == []


def test_batchify_by_fragment_empty_iterable():
    data = []
    batches = list(batchify_by_fragment(data))
    assert batches == []


def test_stat_batchify_empty_iterable():
    data = []
    batch_fn = stat_batchify("length")
    batches = list(batch_fn(data, batch_size=5))
    assert batches == []


def test_batchify_invalid_sentinel_mode():
    data = [1, 2, 3]
    with pytest.raises(AssertionError):
        list(batchify(data, batch_size=2, sentinel_mode="invalid_mode"))
