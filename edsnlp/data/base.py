from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from edsnlp.core.lazy_collection import LazyCollection

from .converters import get_dict2doc_converter, get_doc2dict_converter


class BaseReader:
    """
    The BaseReader servers as a base class for all readers. It expects two methods:

    - `read_main` method which is called in the main process and should return a
        generator of fragments (like filenames) with their estimated size (number of
        documents)
    - `read_worker` method which is called in the worker processes and receives
        batches of fragments and should return a list of dictionaries (one per
        document), ready to be converted to a Doc object by the converter.

    Additionally, the subclass should define a `DATA_FIELDS` class attribute which
    contains the names of all attributes that should not be copied when the reader is
    copied to the worker processes. This is useful for example when the reader holds a
    reference to a large object like a DataFrame that should not be copied to the
    worker processes.
    """

    DATA_FIELDS = ()

    def read_main(self) -> Iterable[Tuple[Any, int]]:
        raise NotImplementedError()

    def read_worker(self, fragment: Iterable[Any]) -> Iterable[Dict]:
        raise NotImplementedError()

    def worker_copy(self):
        # new reader without data, this will not call __init__ since we use __dict__
        # to set the data
        reader = self.__class__.__new__(self.__class__)
        state = {
            k: v
            for k, v in self.__dict__.items()
            if k not in self.__class__.DATA_FIELDS
        }
        reader.__dict__ = state
        return reader


T = TypeVar("T")


class BaseWriter:
    def write_worker(self, records: Sequence[Any]) -> T:
        raise NotImplementedError()

    def write_main(self, fragments: Iterable[T]):
        raise NotImplementedError()

    def finalize(self):
        return None, 0


class IterableReader(BaseReader):
    DATA_FIELDS = ("data",)

    def __init__(self, data: Iterable):
        self.data = data

        super().__init__()

    def read_main(self) -> Iterable[Tuple[Any, int]]:
        return ((item, 1) for item in self.data)

    def read_worker(self, fragments):
        return [task for task in fragments]


def from_iterable(
    data: Iterable,
    converter: Union[str, Callable] = None,
    **kwargs,
) -> LazyCollection:
    """
    The IterableReader (or `edsnlp.data.from_iterable`) reads a list of Python objects (
    texts, dictionaries, ...) and yields documents by passing them through the
    `converter` if given, or returns them as is.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)
    doc_iterator = edsnlp.data.from_iterable([{...}], nlp=nlp, converter=...)
    annotated_docs = nlp.pipe(doc_iterator)
    ```

    !!! note "Generator vs list"

        `edsnlp.data.from_iterable` returns a
        [LazyCollection][edsnlp.core.lazy_collection.LazyCollection].
        To iterate over the documents multiple times efficiently or to access them by
        index, you must convert it to a list

        ```{ .python .no-check }
        docs = list(edsnlp.data.from_iterable([{...}], converter=...)
        ```

    Parameters
    ----------
    data: Iterable
        The data to read
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the JSON rows of the data source to Doc objects
    kwargs:
        Additional keyword arguments to pass to the converter. These are documented
        on the [Converters](/data/converters) page.

    Returns
    -------
    LazyCollection
    """
    data = LazyCollection(reader=IterableReader(data))
    if converter:
        converter, kwargs = get_dict2doc_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)
    return data


def to_iterable(
    data: Union[Any, LazyCollection],
    converter: Optional[Union[str, Callable]] = None,
    **kwargs,
):
    """
    `edsnlp.data.to_items` returns an iterator of documents, as converted by the
    `converter`. In comparison to just iterating over a LazyCollection, this will
    also apply the `converter` to the documents, which can lower the data transfer
    overhead when using multiprocessing.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)

    doc = nlp("My document with entities")

    edsnlp.data.to_items([doc], converter="omop")
    ```

    Parameters
    ----------
    data: Union[Any, LazyCollection],
        The data to write (either a list of documents or a LazyCollection).
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the documents to dictionary objects.
    kwargs:
        Additional keyword arguments passed to the converter. These are documented
        on the [Converters](/data/converters) page.
    """
    data = LazyCollection.ensure_lazy(data)
    if converter:
        converter, kwargs = get_doc2dict_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)

    return data
