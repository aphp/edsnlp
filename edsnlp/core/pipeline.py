import contextlib
import functools
import importlib
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import warnings
from enum import Enum
from inspect import Parameter, signature
from pathlib import Path
from types import FunctionType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Container,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import pkg_resources
import requests
import spacy
import srsly
from confit import Config
from confit.errors import ConfitValidationError, patch_errors
from confit.utils.xjson import Reference
from spacy.language import BaseDefaults
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.util import get_lang_class
from spacy.vocab import Vocab, create_vocab
from typing_extensions import Literal, Self

from ..core.registries import PIPE_META, CurriedFactory, FactoryMeta, registry
from ..utils.collections import (
    FrozenDict,
    FrozenList,
    FrozenNamespace,
    batch_compress_dict,
    decompress_dict,
)
from ..utils.typing import AsList, Validated
from .stream import Stream

if TYPE_CHECKING:
    import torch

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

import edsnlp
from edsnlp.utils.collections import flatten

EMPTY_LIST = FrozenList()
FORBIDDEN_AUTO_HF_OWNERS = {
    "models",
    "artifacts",
    "model-last",
    "model-best",
    ".",
    "..",
}


class CacheEnum(str, Enum):
    preprocess = "preprocess"
    collate = "collate"
    forward = "forward"


Pipe = TypeVar("Pipe", bound=Callable[[Doc], Doc])


@registry.core.register("base")
class Pipeline(Validated):
    """
    New pipeline to use as a drop-in replacement for spaCy's pipeline.
    It uses PyTorch as the deep-learning backend and allows components to share
    subcomponents.

    See the documentation for more details.
    """

    def __init__(
        self,
        lang: str,
        create_tokenizer: Optional[Callable[[Self], Tokenizer]] = None,
        vocab: Union[bool, Vocab] = True,
        batch_size: Optional[int] = None,
        vocab_config: Type[BaseDefaults] = None,
        meta: Dict[str, Any] = None,
        pipeline: Optional[Sequence[str]] = None,
        components: Dict[str, Any] = {},
        disable: AsList[str] = EMPTY_LIST,
        enable: AsList[str] = EMPTY_LIST,
        exclude: AsList = EMPTY_LIST,
    ):
        """
        Parameters
        ----------
        lang: str
            Language code
        create_tokenizer: Optional[Callable[[Self], Tokenizer]]
            Function that creates a tokenizer for the pipeline
        vocab: Union[bool, Vocab]
            Whether to create a new vocab or use an existing one
        vocab_config: Type[BaseDefaults]
            Configuration for the vocab
        meta: Dict[str, Any]
            Meta information about the pipeline
        """
        spacy_blank_cls = get_lang_class(lang)

        self.Defaults = spacy_blank_cls.Defaults
        if batch_size is not None:
            warnings.warn(
                "The 'batch_size' argument is deprecated. Use the 'batch_size' "
                "argument in `stream.map_pipeline` instead.",
                DeprecationWarning,
            )
        self.batch_size = batch_size
        if (vocab is not True) and (vocab_config is not None):
            raise ValueError(
                "You must specify either vocab or vocab_config but not both."
            )
        if vocab is True:
            vocab = create_vocab(lang, vocab_config or BaseDefaults)

        self.vocab = vocab

        if create_tokenizer is None:
            create_tokenizer = Config.resolve(
                spacy_blank_cls.default_config.merge(spacy_blank_cls.Defaults.config)[
                    "nlp"
                ]["tokenizer"]
            )

        self._tokenizer_config = Config.serialize(create_tokenizer)
        self.tokenizer = create_tokenizer(self)
        self._components: List[Tuple[str, Pipe]] = []
        self._disabled: List[str] = []
        self._path: Optional[Path] = None
        self.meta = dict(meta) if meta is not None else {}
        self.lang: str = lang
        self._cache: Optional[Dict] = None

        pipeline = list(components) if pipeline is None else pipeline
        self._add_pipes(pipeline, components, exclude, enable, disable)

    @property
    def pipeline(self) -> List[Tuple[str, Pipe]]:
        return FrozenList(self._components)

    @property
    def pipe_names(self) -> List[str]:
        return FrozenList([name for name, _ in self._components])

    @property
    def pipes(self):
        return FrozenNamespace({name: pipe for name, pipe in self._components})

    component_names = pipe_names

    def get_pipe(self, name: str) -> Pipe:
        """
        Get a component by its name.

        Parameters
        ----------
        name: str
            The name of the component to get.

        Returns
        -------
        Pipe
        """
        for n, pipe in self.pipeline:
            if n == name:
                return pipe
        raise ValueError(f"Pipe {repr(name)} not found in pipeline.")

    def has_pipe(self, name: str) -> bool:
        """
        Check if a component exists in the pipeline.

        Parameters
        ----------
        name: str
            The name of the component to check.

        Returns
        -------
        bool
        """
        return any(n == name for n, _ in self.pipeline)

    def create_pipe(
        self,
        factory: str,
        name: str,
        config: Dict[str, Any] = None,
    ) -> Pipe:
        """
        Create a component from a factory name.

        Parameters
        ----------
        factory: str
            The name of the factory to use
        name: str
            The name of the component
        config: Dict[str, Any]
            The config to pass to the factory

        Returns
        -------
        Pipe
        """
        try:
            pipe = Config(
                {
                    "@factory": factory,
                    **(config if config is not None else {}),
                }
            ).resolve(registry=registry)
            if isinstance(pipe, CurriedFactory):
                if name is None:
                    name = signature(pipe.factory).parameters.get("name").default
                if name is None or name == Parameter.empty:
                    name = factory
                pipe = pipe.instantiate(nlp=self, path=(name,))
        except ConfitValidationError as e:
            raise e.with_traceback(None)
        return pipe

    def add_pipe(
        self,
        factory: Union[str, Pipe],
        first: bool = False,
        before: Optional[str] = None,
        after: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Pipe:
        """
        Add a component to the pipeline.

        Parameters
        ----------
        factory: Union[str, Pipe]
            The name of the component to add or the component itself
        name: Optional[str]
            The name of the component. If not provided, the name of the component
            will be used if it has one (.name), otherwise the factory name will be used.
        first: bool
            Whether to add the component to the beginning of the pipeline. This argument
            is mutually exclusive with `before` and `after`.
        before: Optional[str]
            The name of the component to add the new component before. This argument is
            mutually exclusive with `after` and `first`.
        after: Optional[str]
            The name of the component to add the new component after. This argument is
            mutually exclusive with `before` and `first`.
        config: Dict[str, Any]
            The arguments to pass to the component factory.

            Note that instead of replacing arguments with the same keys, the config
            will be merged with the default config of the component. This means that
            you can override specific nested arguments without having to specify the
            entire config.

        Returns
        -------
        Pipe
            The component that was added to the pipeline.
        """
        if isinstance(factory, str):
            pipe = self.create_pipe(factory, name, config)
            name = name or getattr(pipe, "name", factory)
        else:
            if config is not None:
                raise ValueError(
                    "Can't pass config or name with an instantiated component",
                )
            if isinstance(factory, CurriedFactory):
                name = name or factory.kwargs.get("name")
                factory = factory.instantiate(nlp=self, path=(name,))

            pipe = factory
            name = name or getattr(pipe, "name", None)
            if name is None:
                raise ValueError(
                    "The component does not have a name, so you must provide one",
                )
        if hasattr(pipe, "name") and name != pipe.name:
            pipe.name = name
        assert sum([before is not None, after is not None, first]) <= 1, (
            "You can only use one of before, after, or first",
        )
        insertion_idx = (
            0
            if first
            else self.pipe_names.index(before)
            if before is not None
            else self.pipe_names.index(after) + 1
            if after is not None
            else len(self._components)
        )
        self._components.insert(insertion_idx, (name, pipe))
        return pipe

    def get_pipe_meta(self, name: str) -> FactoryMeta:
        """
        Get the meta information for a component.

        Parameters
        ----------
        name: str
            The name of the component to get the meta for.

        Returns
        -------
        Dict[str, Any]
        """
        pipe = self.get_pipe(name)
        return PIPE_META.get(pipe, {})

    def make_doc(self, text: str) -> Doc:
        """
        Create a Doc from text.

        Parameters
        ----------
        text: str
            The text to create the Doc from.

        Returns
        -------
        Doc
        """
        return self.tokenizer(text)

    def _ensure_doc(self, text: Union[str, Doc]) -> Doc:
        """
        Ensure that the input is a Doc.

        Parameters
        ----------
        text: Union[str, Doc]
            The text to create the Doc from, or a Doc.

        Returns
        -------
        Doc
        """
        return text if isinstance(text, Doc) else self.make_doc(text)

    def __call__(self, text: Union[str, Doc]) -> Doc:
        """
        Apply each component successively on a document.

        Parameters
        ----------
        text: Union[str, Doc]
            The text to create the Doc from, or a Doc.

        Returns
        -------
        Doc
        """
        doc = self._ensure_doc(text)

        with self.cache():
            for name, pipe in self.pipeline:
                if name in self._disabled:
                    continue
                # This is a hack to get around the ambiguity
                # between the __call__ method of Pytorch modules
                # and the __call__ methods of spacy components
                if hasattr(pipe, "batch_process"):
                    doc = next(iter(pipe.batch_process([doc])))
                else:
                    doc = pipe(doc)

        return doc

    def pipe(
        self,
        inputs: Union[Iterable, Stream],
        n_process: int = None,
        **kwargs,
    ) -> Stream:
        """
        Process a stream of documents by applying each component successively on
        batches of documents.

        Parameters
        ----------
        inputs: Iterable[Union[str, Doc]]
            The inputs to create the Docs from, or Docs directly.
        n_process: int
            Deprecated. Use the ".set_processing(num_cpu_workers=n_process)" method
            on the returned data stream instead.
            The number of parallel workers to use. If 0, the operations will be
            executed sequentially.

        Returns
        -------
        Stream
        """

        stream = edsnlp.data.from_iterable(inputs)
        stream = stream.map_pipeline(self, **kwargs)
        if n_process is not None:
            stream = stream.set_processing(num_cpu_workers=n_process)

        return stream

    @contextlib.contextmanager
    def cache(self):
        """
        Enable caching for all (trainable) components in the pipeline
        """
        for name, pipe in self.torch_components():
            pipe.enable_cache()
        yield
        for name, pipe in self.torch_components():
            pipe.disable_cache()

    def torch_components(
        self, disable: Container[str] = ()
    ) -> Iterable[Tuple[str, "edsnlp.core.torch_component.TorchComponent"]]:
        """
        Yields components that are PyTorch modules.

        Parameters
        ----------
        disable: Container[str]
            The names of disabled components, which will be skipped.

        Returns
        -------
        Iterable[Tuple[str, 'edsnlp.core.torch_component.TorchComponent']]
        """
        for name, pipe in self.pipeline:
            if name not in disable and hasattr(pipe, "forward"):
                yield name, pipe

    def connected_pipes_names(self) -> List[List[str]]:
        """
        Returns a list of lists of connected components in the pipeline,
        i.e. components that share at least one parameter.

        Returns
        -------
        List[List[str]]
        """
        pipe_to_params = {}
        for name, pipe in self.torch_components():
            pipe_to_params[name] = {id(p) for p in pipe.parameters()}
        remaining_pipes = list(pipe_to_params)
        results = []
        while len(remaining_pipes):
            current = [remaining_pipes.pop(0)]
            i = 0
            while i < len(current):
                a = current[i]
                i += 1
                for j, b in enumerate(list(remaining_pipes)):
                    if a is not b and pipe_to_params[a] & pipe_to_params[b]:
                        current.append(b)
                        remaining_pipes[j] = None
                remaining_pipes = [p for p in remaining_pipes if p is not None]

            results.append(current)
        return results

    def post_init(self, data: Iterable[Doc], exclude: Optional[Set] = None):
        """
        Completes the initialization of the pipeline by calling the post_init
        method of all components that have one.
        This is useful for components that need to see some data to build
        their vocabulary, for instance.

        Parameters
        ----------
        data: Iterable[Doc]
            The documents to use for initialization.
            Each component will not necessarily see all the data.
        exclude: Optional[Container]
            Components to exclude from post initialization on data
        """
        exclude = set() if exclude is None else set(exclude)
        for name, pipe in self._components:
            if (
                hasattr(pipe, "post_init")
                and name not in self.disabled
                and name not in exclude
            ):
                pipe.post_init(data, exclude=exclude)

    @classmethod
    def from_config(
        cls,
        config: Union[Dict[str, Any], "Pipeline"] = {},
        *,
        vocab: Union[Vocab, bool] = True,
        disable: Union[str, Iterable[str]] = EMPTY_LIST,
        enable: Union[str, Iterable[str]] = EMPTY_LIST,
        exclude: Union[str, Iterable[str]] = EMPTY_LIST,
        meta: Dict[str, Any] = FrozenDict(),
    ):
        """
        Create a pipeline from a config object

        Parameters
        ----------
        config: Config
            The config to use
        vocab: Union["spacy.Vocab", bool]
            The spaCy vocab to use. If True, a new vocab will be created
        disable: Union[str, Iterable[str]]
            Components to disable
        enable: Union[str, Iterable[str]]
            Components to enable
        exclude: Union[str, Iterable[str]]
            Components to exclude
        meta: Dict[str, Any]
            Metadata to add to the pipeline

        Returns
        -------
        Pipeline
        """
        if isinstance(config, Pipeline):  # pragma: no cover
            return config
        root_config = config.copy()

        if "nlp" in config:
            if isinstance(config["nlp"], Pipeline):  # pragma: no cover
                return config["nlp"]
            if "components" in config and "components" not in config["nlp"]:
                config["nlp"]["components"] = Reference("components")
            config = config["nlp"]

        config = Config(config).resolve(root=root_config, registry=registry)
        if isinstance(config, Pipeline):  # pragma: no cover
            return config
        config = dict(config)
        components = config.pop("components", {})
        pipeline = config.pop("pipeline", list(components.keys()))
        tokenizer = config.pop("tokenizer", None)
        exclude = list(flatten([*config.pop("exclude", ()), exclude]))
        enable = list(flatten([*config.pop("enable", ()), enable]))
        disable = list(flatten([*config.pop("disable", ()), disable]))

        nlp = Pipeline(
            vocab=vocab,
            create_tokenizer=tokenizer,
            meta=meta,
            pipeline=pipeline,
            components=components,
            exclude=exclude,
            enable=enable,
            disable=disable,
            **config,
        )
        return nlp

    def _add_pipes(
        self,
        pipeline: Sequence[str],
        components: Dict[str, CurriedFactory],
        exclude: Container[str],
        enable: Container[str],
        disable: Container[str],
    ):
        try:
            components = CurriedFactory.instantiate(components, nlp=self)
        except ConfitValidationError as e:
            e = ConfitValidationError(
                e.raw_errors,
                model=self.__class__,
                name=self.__class__.__module__ + "." + self.__class__.__qualname__,
            )
            e.raw_errors = patch_errors(e.raw_errors, ("components",))
            raise e

        for name in pipeline:
            if name in exclude:
                continue
            if name not in components:
                raise ValueError(f"Component {repr(name)} not found in config")
            self.add_pipe(components[name], name=name)

        # Set of components name if it's in the disable list
        # or if it's not in the enable list and the enable list is not empty
        self._disabled = [
            name
            for name in self.pipe_names
            if name in disable or (enable and name not in enable)
        ]

    @property
    def disabled(self):
        """
        The names of the disabled components
        """
        return FrozenList(self._disabled)

    @classmethod
    def __get_validators__(cls):
        """
        Pydantic validators generator
        """
        yield cls.validate

    @classmethod
    def validate(cls, v, config=None):
        """
        Pydantic validator, used in the `validate_arguments` decorated functions
        """
        if isinstance(v, dict):
            return cls.from_config(v)
        if not isinstance(v, cls):
            raise ValueError("input is not a Pipeline or config dict")
        return v

    @property
    def preprocess(self):
        compressor = batch_compress_dict()
        disabled = list(self._disabled)

        @functools.wraps(self._preprocess)
        def preprocess(doc, supervision: bool = False, compress: bool = True):
            prep = self._preprocess(doc, supervision, disabled)
            if compress:
                prep = compressor(prep)
            return prep

        return preprocess

    def _preprocess(
        self,
        doc: Doc,
        supervision: bool = False,
        disabled: Container[str] = (),
    ):
        """
        Run the preprocessing methods of each component in the pipeline
        on a document and returns a dictionary containing the results, with the
        component names as keys.

        Parameters
        ----------
        doc: InputT
            The document to preprocess
        supervision: bool
            Whether to include supervision information in the preprocessing

        Returns
        -------
        Dict[str, Any]
        """
        prep = {}
        with self.cache():
            for name, component in self.pipeline:
                if name not in disabled:
                    prep_comp = (
                        component.preprocess_supervised(doc)
                        if supervision and hasattr(component, "preprocess_supervised")
                        else component.preprocess(doc)
                        if hasattr(component, "preprocess")
                        else None
                    )
                    if prep_comp is not None:
                        prep[name] = prep_comp
        return prep

    def preprocess_many(self, docs: Iterable[Doc], compress=True, supervision=True):
        """
        Runs the preprocessing methods of each component in the pipeline on
        a collection of documents and returns an iterable of dictionaries containing
        the results, with the component names as keys.

        Parameters
        ----------
        docs: Iterable[InputT]
        compress: bool
            Whether to deduplicate identical preprocessing outputs of the results
            if multiple documents share identical subcomponents. This step is required
            to enable the cache mechanism when training or running the pipeline over a
            tabular datasets such as pyarrow tables that do not store referential
            equality information.
        supervision: bool
            Whether to include supervision information in the preprocessing

        Returns
        -------
        Stream
        """
        res = Stream.ensure_stream(docs)
        res = res.map(functools.partial(self.preprocess, supervision=supervision))
        return res

    def collate(
        self,
        batch: Union[Iterable[Dict[str, Any]], Dict[str, Any]],
        device: Optional[Union[str, "torch.device"]] = None,
    ):
        """
        Collates a batch of preprocessed samples into a single (maybe nested)
        dictionary of tensors by calling the collate method of each component.

        Parameters
        ----------
        batch: Union[Iterable[Dict[str, Any]], Dict[str, Any]]
            The batch of preprocessed samples
        device: Optional[Union[str, "torch.device"]]
            Should we move the tensors to a device, if so, which one?

        Returns
        -------
        Dict[str, Any]
            The collated batch
        """
        batch = decompress_dict(batch)
        with self.cache():
            for name, component in self.pipeline:
                if name in batch:
                    component_inputs = batch[name]
                    batch[name] = component.collate(component_inputs)
                    if device is not None:
                        batch[name] = component.batch_to_device(batch[name], device)
        return batch

    def parameters(self):
        """Returns an iterator over the Pytorch parameters of the components in the
        pipeline"""
        return (p for n, p in self.named_parameters())

    def named_parameters(self):
        """Returns an iterator over the Pytorch parameters of the components in the
        pipeline"""
        seen = set()
        for name, component in self.pipeline:
            if hasattr(component, "named_parameters"):
                for param_name, param in component.named_parameters():
                    if param in seen:
                        continue
                    seen.add(param)
                    yield f"{name}.{param_name}", param

    def to(self, device: Union[str, Optional["torch.device"]] = None):  # noqa F821
        """Moves the pipeline to a given device"""
        for name, component in self.torch_components():
            component.to(device)
        return self

    def train(self, mode=True):
        """
        Enables training mode on pytorch modules

        Parameters
        ----------
        mode: bool
            Whether to enable training or not
        """

        class context:
            def __enter__(self):
                pass

            def __exit__(ctx_self, type, value, traceback):
                for name, proc in self.torch_components():
                    proc.train(was_training[name])

        was_training = {name: proc.training for name, proc in self.torch_components()}
        for name, proc in self.torch_components():
            proc.train(mode)

        return context()

    def to_disk(
        self, path: Union[str, Path], *, exclude: Optional[Set[str]] = None
    ) -> None:
        """
        Save the pipeline to a directory.

        Parameters
        ----------
        path: Union[str, Path]
            The path to the directory to save the pipeline to. Every component will be
            saved to separated subdirectories of this directory, except for tensors
            that will be saved to a shared files depending on the references between
            the components.
        exclude: Sequence[str]
            The names of the components, or attributes to exclude from the saving
            process.
            By default, the vocabulary is excluded since it may contain personal
            identifiers and can be rebuilt during inference.
        """
        exclude = {"vocab"} if exclude is None else exclude

        path = Path(path) if isinstance(path, str) else path

        if (
            os.path.exists(path)
            and os.listdir(path)
            and not os.path.exists(path / "config.cfg")
        ):
            raise Exception(
                f"The directory {path} already exists and doesn't appear to be a "
                "saved pipeline (missing config.cfg). Please erase it manually or "
                "choose a different directory."
            )
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

        if "tokenizer" not in exclude:
            self.tokenizer.to_disk(path / "tokenizer", exclude=["vocab"])
        if "meta" not in exclude:
            srsly.write_json(path / "meta.json", self.meta)
        if "vocab" not in exclude:
            self.vocab.to_disk(path / "vocab")

        pwd = os.getcwd()
        overrides = {"components": {}}
        try:
            os.chdir(path)
            for pipe_name, pipe in self._components:
                if hasattr(pipe, "to_disk") and pipe_name not in exclude:
                    pipe_overrides = pipe.to_disk(Path(pipe_name), exclude=exclude)
                    overrides["components"][pipe_name] = pipe_overrides
        finally:
            os.chdir(pwd)

        config = self.config.merge(overrides)

        if "config" not in exclude:
            config.to_disk(path / "config.cfg")

    def from_disk(
        self,
        path: Union[str, Path],
        *,
        exclude: Optional[Union[str, Sequence[str]]] = None,
        device: Optional[Union[str, "torch.device"]] = "cpu",  # noqa F821
    ) -> "Pipeline":
        """
        Load the pipeline from a directory. Components will be updated in-place.

        Parameters
        ----------
        path: Union[str, Path]
            The path to the directory to load the pipeline from
        exclude: Optional[Union[str, Sequence[str]]]
            The names of the components, or attributes to exclude from the loading
            process.
        device: Optional[Union[str, "torch.device"]]
            Device to use when loading the tensors
        """

        def deserialize_meta(path: Path) -> None:
            if path.exists():
                data = srsly.read_json(path)
                self.meta.update(data)
                # self.meta always overrides meta["vectors"] with the metadata
                # from self.vocab.vectors, so set the name directly
                self.vocab.vectors.name = data.get("vectors", {}).get("name")

        def deserialize_vocab(path: Path) -> None:
            if path.exists():
                self.vocab.from_disk(path, exclude=exclude)

        exclude = (
            set()
            if exclude is None
            else {exclude}
            if isinstance(exclude, str)
            else set(exclude)
        )

        path = (Path(path) if isinstance(path, str) else path).absolute()
        if (path / "meta.json").exists() and "meta" not in exclude:
            deserialize_meta(path / "meta.json")
        if (path / "vocab").exists() and "vocab" not in exclude:
            deserialize_vocab(path / "vocab")
        if (path / "tokenizer").exists() and "tokenizer" not in exclude:
            self.tokenizer.from_disk(path / "tokenizer", exclude=["vocab"])

        pwd = os.getcwd()
        try:
            os.chdir(path)
            for name, proc in self._components:
                if hasattr(proc, "from_disk") and name not in exclude:
                    proc.from_disk(Path(name), exclude=exclude)
                # Convert to list here in case exclude is (default) tuple
                exclude.add(name)
        finally:
            os.chdir(pwd)

        self._path = path  # type: ignore[assignment]
        return self

    # override config property getter to remove "factory" key from components
    @property
    def cfg(self) -> Config:
        """
        Returns the config of the pipeline, including the config of all components.
        Updated from spacy to allow references between components.
        """
        return Config(
            {
                "lang": self.lang,
                "pipeline": list(self.pipe_names),
                "components": {key: component for key, component in self._components},
                "tokenizer": self._tokenizer_config,
            }
        )

    @property
    def config(self) -> Config:
        config = Config({"nlp": self.cfg.copy()})
        config["components"] = config["nlp"].pop("components")
        return config.serialize()

    def select_pipes(
        self,
        *,
        disable: Optional[Union[str, Iterable[str]]] = None,
        enable: Optional[Union[str, Iterable[str]]] = None,
    ):
        """
        Temporarily disable and enable components in the pipeline.

        Parameters
        ----------
        disable: Optional[Union[str, Iterable[str]]]
            The name of the component to disable, or a list of names.
        enable: Optional[Union[str, Iterable[str]]]
            The name of the component to enable, or a list of names.
        """

        class context:
            def __enter__(self):
                pass

            def __exit__(ctx_self, type, value, traceback):
                self._disabled = disabled_before

        disable = [disable] if isinstance(disable, str) else disable
        pipe_names = set(self.pipe_names)
        if enable is not None:
            enable = [enable] if isinstance(enable, str) else enable
            if set(enable) - pipe_names:
                raise ValueError(
                    "Enabled pipes {} not found in pipeline.".format(
                        sorted(set(enable) - pipe_names)
                    )
                )
            to_disable = [pipe for pipe in self.pipe_names if pipe not in enable]
            # raise an error if the enable and disable keywords are not consistent
            if disable is not None and disable != to_disable:
                raise ValueError("Inconsistent values for `enable` and `disable`")
            disable = to_disable

        disable = disable or ()
        if set(disable) - pipe_names:
            raise ValueError(
                "Disabled pipes {} not found in pipeline.".format(
                    sorted(set(disable) - pipe_names)
                )
            )

        disabled_before = self._disabled
        self._disabled = disable
        return context()

    def package(
        self,
        name: Optional[str] = None,
        root_dir: Union[str, Path] = ".",
        build_dir: Optional[Union[str, Path]] = None,
        dist_dir: Union[str, Path] = "dist",
        artifacts_name: str = "artifacts",
        check_dependencies: bool = False,
        project_type: Optional[Literal["poetry", "setuptools"]] = None,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = {},
        distributions: Optional[AsList[Literal["wheel", "sdist"]]] = ["wheel"],
        config_settings: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        isolation: bool = True,
        skip_build_dependency_check: bool = False,
        readme_replacements: Dict[str, str] = {},
    ):
        from edsnlp.package import package

        return package(
            pipeline=self,
            name=name,
            root_dir=root_dir,
            build_dir=build_dir,
            dist_dir=dist_dir,
            artifacts_name=artifacts_name,
            check_dependencies=check_dependencies,
            project_type=project_type,
            version=version,
            metadata=metadata,
            distributions=distributions,
            config_settings=config_settings,
            isolation=isolation,
            skip_build_dependency_check=skip_build_dependency_check,
            readme_replacements=readme_replacements,
        )

    if TYPE_CHECKING:
        from edsnlp.package import package as package

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pipe_meta"] = [PIPE_META.get(pipe, {}) for _, pipe in self.pipeline]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for (name, pipe), meta in zip(self.pipeline, state["_pipe_meta"]):
            PIPE_META[pipe] = meta
        del state["_pipe_meta"]

    def __repr__(self):
        pipes_str = ""
        for name, pipe in self.pipeline:
            if pipes_str:
                pipes_str += ","
            try:
                factory_name = Config.serialize(pipe)["@factory"]
            except KeyError:
                if isinstance(pipe, FunctionType):
                    factory_name = pipe
                else:
                    factory_name = pipe.__class__.__name__
            disabled = " [disabled] " if name in self._disabled else " "
            pipe_str = f'"{name}":{disabled}{factory_name}'
            pipes_str += f"\n  {pipe_str}"
        if pipes_str:
            pipes_str += "\n"
        return f"Pipeline(lang={self.lang}, pipes={{{pipes_str}}})"


def blank(
    lang: str,
    config: Union[Dict[str, Any], Config] = {},
):
    """
    Loads an empty EDS-NLP Pipeline, similarly to `spacy.blank`. In addition to
    standard components, this pipeline supports EDS-NLP trainable torch components.

    Examples
    --------
    ```python
    import edsnlp, edsnlp.pipes as eds

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(eds.covid())
    ```

    Parameters
    ----------
    lang: str
        Language ID, e.g. "en", "fr", "eds", etc.
    config: Union[Dict[str, Any], Config]
        The config to use for the pipeline

    Returns
    -------
    Pipeline
        The new empty pipeline instance.
    """
    # Check if language is registered / entry point is available
    config_lang = (
        config["nlp"].get("lang")
        if isinstance(config.get("nlp"), dict)
        else config.get("lang")
    )
    if config_lang is not None and config_lang != lang:
        raise ValueError(
            "The language specified in the config does not match the lang argument."
        )
    return Pipeline.from_config({"lang": lang, **config})


@registry.core.register("load")
def load(
    model: Union[str, Path, Config],
    overrides: Optional[Dict[str, Any]] = None,
    *,
    exclude: Optional[AsList[str]] = EMPTY_LIST,
    disable: Optional[AsList[str]] = EMPTY_LIST,
    enable: Optional[AsList[str]] = EMPTY_LIST,
    auto_update: bool = False,
    install_dependencies: bool = False,
    **kwargs,
):
    """
    Load a pipeline from a config file or a directory.

    Examples
    --------

    ```{ .python .no-check }
    import edsnlp

    nlp = edsnlp.load(
        "path/to/config.cfg",
        overrides={"components": {"my_component": {"arg": "value"}}},
    )
    ```

    Parameters
    ----------
    model: Union[str, Path, Config]
        The config to use for the pipeline, or the path to a config file or a directory.
    overrides: Optional[Dict[str, Any]]
        Overrides to apply to the config when loading the pipeline. These are the
        same parameters as the ones used when initializing the pipeline.
    exclude: Optional[Union[str, Iterable[str]]]
        The names of the components, or attributes to exclude from the loading
        process. :warning: The `exclude` argument will be mutated in place.
    auto_update: bool
        When installing a pipeline from the Hugging Face Hub, whether to automatically
        try to update the model, even if a local version is found. Only applies when
        loading from the Hugging Face Hub.
    install_dependencies: bool
        When installing a pipeline from the Hugging Face Hub, whether to install the
        dependencies of the model if they are not already installed. Only applies when
        loading from the Hugging Face Hub.

    Returns
    -------
    Pipeline
    """
    base_exc = None
    error = (
        "The load function expects either :\n"
        "- a confit Config object\n"
        "- the path of a config file (.cfg file)\n"
        "- the path of a trained model\n"
        "- the name of an installed pipeline package\n"
        "- or a model on the huggingface hub if edsnlp[ml] has been installed\n"
        f"but got {model!r} which is neither"
    )
    pipe_selection = {
        "exclude": exclude,
        "disable": disable,
        "enable": enable,
    }
    pipe_selection = {k: v for k, v in pipe_selection.items() if v is not EMPTY_LIST}
    if isinstance(model, (str, Path)):
        path = Path(model)
        is_dir = path.is_dir()
        is_config = path.is_file() and path.suffix in (".cfg", ".yml", ".yaml")
        try:
            module = importlib.import_module(model)
            is_package = True
        except (ImportError, AttributeError, TypeError):
            module = None
            is_package = False
        if is_dir and is_package:
            warnings.warn(
                "The path provided is both a directory and a package : edsnlp will "
                "load the package. To load from the directory instead, please pass the "
                f'path as "./{path}" instead.'
            )
        if is_dir:
            path = (Path(path) if isinstance(path, str) else path).absolute()
            config = Config.from_disk(path / "config.cfg")
            if overrides:
                config = config.merge(overrides)
            pwd = os.getcwd()
            try:
                os.chdir(path)
                nlp = Pipeline.from_config(config, **pipe_selection)
                nlp.from_disk(path, exclude=exclude)
                nlp.train(False)
            finally:
                os.chdir(pwd)
            return nlp
        elif is_config:
            model = Config.from_disk(path)
        elif is_package:
            # Load as package
            available_kwargs = {"overrides": overrides, **pipe_selection}
            signature_kwargs = signature(module.load).parameters
            kwargs = {
                name: available_kwargs[name]
                for name in signature_kwargs
                if name in available_kwargs
            }
            return module.load(**kwargs)
        elif (
            isinstance(model, str)
            and len(model.split("/")) == 2
            and model.split("/")[0] not in FORBIDDEN_AUTO_HF_OWNERS
        ):
            try:
                return load_from_huggingface(
                    model,
                    overrides=overrides,
                    auto_update=auto_update,
                    install_dependencies=install_dependencies,
                    **kwargs,
                )
            except (
                ImportError,
                requests.RequestException,
                ValueError,
                FileNotFoundError,
            ) as e:
                base_exc = e

    if not isinstance(model, Config):
        raise ValueError(error) from base_exc

    return Pipeline.from_config(model)


def load_from_huggingface(
    repo_id: str,
    auto_update: bool = False,
    install_dependencies: bool = False,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    **kwargs,
):
    """
    Load a model from the Hugging Face Hub.

    Parameters
    ----------
    repo_id: str
        The repository ID of the model to load (e.g. "username/repo_name").
    auto_update: bool
        Whether to automatically try to update the model, even if a local version
        is found.
    install_dependencies: bool
        Whether to install the dependencies of the model if they are not already
        installed.
    token: Optional[str]
        The Hugging Face Hub API token to use.
    revision: Optional[str]
        The revision of the model to load.
    kwargs: Any
        Additional keyword arguments to pass to the model's `load` method.

    Returns
    -------

    """
    from huggingface_hub import snapshot_download

    owner, model_name = repo_id.split("/")
    module_name = model_name.replace("-", "_")

    assert (
        len(repo_id.split("/")) == 2
    ), "Invalid repo_id format (expected 'owner/repo_name' format)"
    path = None
    mtime = None
    try:
        path = snapshot_download(
            repo_id,
            local_files_only=auto_update,
            token=token,
            revision=revision,
        )
        mtime = max(os.path.getmtime(x) for x in Path(path).rglob("*"))
    except FileNotFoundError:
        pass

    should_install = False
    if path is None or auto_update:
        # Download the snapshot, which is the source distribution of the packaged model
        path = snapshot_download(
            repo_id,
            local_files_only=False,
            token=token,
            revision=revision,
        )
        new_mtime = max(os.path.getmtime(x) for x in Path(path).rglob("*"))
        should_install = new_mtime != mtime

    pip = os.path.join(sysconfig.get_path("scripts"), "pip")
    if should_install or not any(
        p.startswith(module_name) and p.endswith(".dist-info") for p in os.listdir(path)
    ):
        subprocess.run(
            [pip, "install", "-e", path, "--target", path, "--no-deps", "--upgrade"]
        )

    if path not in sys.path:
        sys.path.insert(0, path)

    # Load dependencies
    reqs = importlib_metadata.requires(module_name)
    missing_deps = []
    # Check if the dependencies are installed, with the correct version
    for req in reqs:
        req_without_extra = re.sub(r"\[.*\]", "", req)
        try:
            pkg_resources.require(req_without_extra)
        except (pkg_resources.VersionConflict, pkg_resources.DistributionNotFound):
            missing_deps.append(req)

    if missing_deps:
        if not install_dependencies:
            warnings.warn(
                "Some dependencies could not be satisfied, consider installing them "
                "to use the model:\n"
                f"  pip install {' '.join((repr(str(dep)) for dep in missing_deps))}\n"
                f"or let edsnlp do it \n"
                f"  nlp = edsnlp.load('{repo_id}', install_dependencies=True)\n"
                f"You may need to restart your Python session after the installation.",
                UserWarning,
            )
        else:
            warnings.warn(
                "Installing missing dependencies:\n"
                f"pip install {' '.join((repr(str(dep)) for dep in missing_deps))}",
                UserWarning,
            )
            subprocess.run([pip, "install", *(str(d) for d in missing_deps)])
    module = importlib.import_module(module_name)
    return module.load(**kwargs)


PipelineProtocol = Union[Pipeline, spacy.Language]
