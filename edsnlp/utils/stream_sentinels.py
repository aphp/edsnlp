class StreamSentinel:
    pass


class FragmentEndSentinel(StreamSentinel):
    kind = "fragment"

    def __init__(self, name: str):
        self.name = name


class DatasetEndSentinel(StreamSentinel):
    # Singleton is important since the DatasetEndSentinel object may be passed to
    # other processes, i.e. pickled, depickled, while it should
    # always be the same object.
    kind = "dataset"
    instance = None

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance


DATASET_END_SENTINEL = DatasetEndSentinel()
