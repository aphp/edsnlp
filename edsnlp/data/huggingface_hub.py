import random
import re
import ast
import warnings
from typing import Any, Callable, Iterable, Literal, Optional, Union, Dict
from confit import validate_arguments

from edsnlp import registry
from edsnlp.core.stream import Stream
from edsnlp.data.base import MemoryBasedReader
from edsnlp.data.converters import get_dict2doc_converter
from edsnlp.utils.stream_sentinels import DatasetEndSentinel


class HFDatasetReader(MemoryBasedReader):
    def __init__(
        self,
        dataset: Any,
        shuffle: Union[Literal["dataset"], bool] = False,
        seed: Optional[int] = None,
        loop: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.shuffle = "dataset" if shuffle == "dataset" or shuffle is True else False
        seed = seed if seed is not None else random.getrandbits(32)
        self.rng = random.Random(seed)
        self.loop = loop
        self.emitted_sentinels = {"dataset"}
    
    def read_records(self) -> Iterable[Any]:
        while True:
            data = self.dataset
            if self.shuffle == "dataset":
                try:
                    data = list(self.dataset)
                except Exception:
                    # fallback to non-shuffled iterator
                    data = self.dataset
                else:
                    self.rng.shuffle(data)
            
            # If dataset supports iteration of examples
            try:
                for item in data:
                    yield item
            except TypeError:
                # Not iterable? try indexes
                for i in range(len(data)):
                    yield data[i]

            yield DatasetEndSentinel()
            if not self.loop:
                break
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={object.__repr__(self.dataset)}, "
            f"shuffle={self.shuffle}, "
            f"loop={self.loop})"
        )
            
        
@registry.readers.register("huggingface_hub")
@validate_arguments()
def from_huggingface_hub(
    dataset: Union[str, Any],
    split: Optional[str] = None,
    name: Optional[str] = None,  # Add config/subset name parameter
    shuffle: Union[Literal["dataset"], bool] = False,
    seed: Optional[int] = None,
    loop: bool = False,
    converter: Union[str, Callable] = "hf_ner",
    load_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Stream:
    """
    Load a dataset from the HuggingFace Hub as a Stream.
    
    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)
    doc_iterator = edsnlp.data.from_huggingface_hub(
        "lhoestq/conll2003",
        split="train",
        tag_order=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
        converter="hf_ner",
    )
    annotated_docs = nlp.pipe(doc_iterator)
    ```
    
    Parameters
    ----------
    dataset: Union[str, Any]
        Either a dataset identifier (e.g. "conll2003") or an already loaded
        `datasets.Dataset` / `datasets.IterableDataset` object.
    split: Optional[str]
        Which split to load (e.g. "train"). If None, the default dataset split
        returned by `datasets.load_dataset` is used.
    name: Optional[str]
        Configuration name for datasets with multiple configs (e.g. "en" for 
        a multilingual dataset). Also known as the subset name.
    converter: Optional[Union[str, Callable]]
        Converter(s) to transform dataset dicts to Doc objects. Recommended
        converters are `"hf_ner"` and `"hf_text"`. More information is available
        in the [Converters](/data/converters) page.
    shuffle: Literal['dataset', False] or bool
        Whether to shuffle the dataset before yielding. If True or 'dataset', the
        whole dataset will be materialized and shuffled (may be expensive).
    seed: Optional[int]
        Random seed for shuffling.
    loop: bool
        Whether to loop over the dataset indefinitely.
    load_kwargs: dict
        Dictionary of additional kwargs that will be passed to the 
        `datasets.load_dataset()` method.
    kwargs: dict
        Additional keyword arguments passed to the converter, these are
        documented in the [Converters](/data/converters) page.

    Returns
    -------
    Stream
    """
    load_kwargs = load_kwargs or {}

    try:
        import datasets
    except Exception as e:  # pragma: no cover - dependency handling
        raise ImportError(
            "The 'datasets' library is required to load huggingface datasets. "
            "Install it with `pip install datasets` or with the edsnlp extras: "
            "`pip install 'edsnlp[ml]'`"
        ) from e
    
    # If user passed a dataset identifier string, load it
    ds = dataset
    if isinstance(dataset, str):
        try:
            ds = datasets.load_dataset(dataset, name=name, split=split, **load_kwargs)
        except ValueError as e:
            msg = str(e)
            # Handle datasets that require a config name when none was provided
            if "Config name is missing" in msg and name is None:
                m = re.search(r"available configs: (\[.*\])", msg)
                if m:
                    try:
                        configs = ast.literal_eval(m.group(1))
                        if configs:
                            chosen = configs[0]
                            warnings.warn(
                                f"Dataset {dataset!r} requires a config name; "
                                f"no `name` was provided. Using first available config "
                                f"'{chosen}'. Pass `name` to select another config.",
                                UserWarning,
                            )
                            ds = datasets.load_dataset(dataset, name=chosen, split=split, **load_kwargs)
                        else:
                            raise
                    except Exception:
                        raise ValueError(
                            f"Config name is missing for dataset {dataset!r}. "
                            f"Please pass a `name` among the available configs as reported by the dataset builder."
                        ) from e
                else:
                    raise ValueError(
                        f"Config name is missing for dataset {dataset!r}. "
                        f"Please pass a `name` among the available configs as reported by the dataset builder."
                    ) from e
            else:
                raise
    else:
        # If user passed a (split) name to select
        if split is not None and hasattr(dataset, "select"):
            try:
                ds = dataset[split]
            except Exception:
                pass
    # If no split was provided and the loaded dataset exposes multiple splits
    # (e.g., a `DatasetDict`), pick the 'train' split by default and warn
    # the user to be explicit.
    try:
        if split is None and hasattr(ds, "keys") and "train" in ds.keys():
            warnings.warn(
                f"Dataset {dataset!r} contains multiple splits and no `split` was provided; "
                f"using 'train' by default. Pass `split` to select another split.",
                UserWarning,
            )
            ds = ds["train"]
    except Exception:
        # Be conservative: if detection fails, keep ds as-is
        pass
    # Inspect available columns / features to give better errors and autodetection
    col_names = None
    try:
        if hasattr(ds, "column_names"):
            col_names = list(ds.column_names)
        elif hasattr(ds, "features") and isinstance(ds.features, dict):
            col_names = list(ds.features.keys())
        else:
            # Try to peek the first example
            try:
                first = ds[0]
                if isinstance(first, dict):
                    col_names = list(first.keys())
            except Exception:
                # Could be streaming/iterable without indexing
                try:
                    it = iter(ds)
                    first = next(it)
                    if isinstance(first, dict):
                        col_names = list(first.keys())
                except Exception:
                    col_names = None
    except Exception:
        col_names = None

    # If the user requested the hf_ner converter, ensure required columns exist
    if converter == "hf_ner":
        id_col = kwargs.get("id_column", "id")
        words_col = kwargs.get("words_column", "tokens")
        ner_col = kwargs.get("ner_tags_column", "ner_tags")

        missing = []
        if col_names is not None:
            if id_col not in col_names:
                missing.append(f"`id_column`={id_col}")
            if words_col not in col_names:
                missing.append(f"`words_column`={words_col}")
            if ner_col not in col_names:
                missing.append(f"`ner_tags_column`={ner_col}")

        if col_names is not None and missing:
            # Build helpful message
            raise ValueError(
                "Cannot find these columns in dataset: "
                f"{missing}. "
                + f"Dataset columns are: {col_names}."
                + (
                    " If you intended to process raw text, consider using the 'hf_text' "
                    "converter (pass `converter='hf_text'` and `text_column='<column>'`)."
                )
            )
         
        kwargs["id_column"] = id_col
        kwargs["words_column"] = words_col
        kwargs["ner_tags_column"] = ner_col
        
        
    # If the user requested the hf_text converter, ensure required column exists
    if converter == "hf_text":
        id_col = kwargs.get("id_column", "id")
        text_col = kwargs.get("text_column", "text")
        
        missing = []
        if col_names is not None:
            if id_col not in col_names:
                missing.append(f"`id_column`={id_col}")
            if text_col not in col_names:
                missing.append(f"`text_column`={text_col}")

        if col_names is not None and missing:
            # Build helpful message
            raise ValueError(
                "Cannot find these columns in dataset: "
                f"{missing}. "
                + f"Dataset columns are: {col_names}."
                + (
                    " If you intended to process a NER dataset, consider using the 'hf_ner' "
                    "converter (pass `converter='hf_ner')."
                )
            )
        
        kwargs["id_column"] = id_col
        kwargs["text_column"] = text_col

    reader = HFDatasetReader(ds, shuffle=shuffle, seed=seed, loop=loop)
    stream = Stream(reader=reader)
    
    if converter:
        conv, kwargs = get_dict2doc_converter(converter, kwargs)
        stream = stream.map(conv, kwargs=kwargs)
    
    return stream

@registry.writers.register("huggingface_hub")
def to_huggingface_hub():
    #TODO : implement this ?
    pass
