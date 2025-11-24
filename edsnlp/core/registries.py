import importlib.metadata as importlib_metadata
import inspect
import types
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)
from weakref import WeakKeyDictionary

import catalogue
import confit
import spacy
import spacy.registrations
from confit import Config, RegistryCollection, set_default_registry
from confit.errors import ConfitValidationError, patch_errors
from confit.registry import Draft
from spacy.pipe_analysis import validate_attrs
from spacy.pipeline.factories import register_factories

import edsnlp
from edsnlp.utils.collections import FrozenDict, FrozenList

PIPE_META = WeakKeyDictionary()


def accepted_arguments(
    func: Callable,
    args: Sequence[str],
) -> Sequence[str]:
    """
    Checks that a function accepts a list of keyword arguments

    Parameters
    ----------
    func: Callable[..., T]
        Function to check
    args: Union[str, Sequence[str]]
        Argument or list of arguments to check

    Returns
    -------
    List[str]
    """
    sig = inspect.signature(func)
    has_kwargs = any(
        param.kind == param.VAR_KEYWORD for param in sig.parameters.values()
    )
    if has_kwargs:
        return args
    return [arg for arg in args if arg in sig.parameters]


@dataclass
class FactoryMeta:
    assigns: Iterable[str]
    requires: Iterable[str]
    retokenizes: bool
    default_config: Dict


T = TypeVar("T")


_AUTO_INSTANTIATE_COMPLETE_PIPE_DRAFT = True


class Registry(confit.Registry):
    def get(self, name: str) -> Any:
        if name.startswith("spacy."):
            spacy.registrations.populate_registry()
        return super().get(name)


class DraftPipe(Draft[T]):
    def __init__(self, func, kwargs):
        super().__init__(func, kwargs)
        self.instantiated = None
        self.error = None

    @staticmethod
    @contextmanager
    def disable_auto_instantiation():
        global _AUTO_INSTANTIATE_COMPLETE_PIPE_DRAFT
        old_value = _AUTO_INSTANTIATE_COMPLETE_PIPE_DRAFT
        _AUTO_INSTANTIATE_COMPLETE_PIPE_DRAFT = False
        try:
            yield
        finally:
            _AUTO_INSTANTIATE_COMPLETE_PIPE_DRAFT = old_value

    def maybe_nlp(self) -> Union["DraftPipe", Any]:
        """
        If the factory requires an nlp argument and the user has explicitly
        provided it (this is unusual, we usually expect the factory to be
        instantiated via add_pipe, or a config), then we should instantiate
        it.

        Returns
        -------
        Union["PartialFactory", Any]
        """
        from edsnlp.core.pipeline import Pipeline, PipelineProtocol

        sig = inspect.signature(self._func)
        if (
            _AUTO_INSTANTIATE_COMPLETE_PIPE_DRAFT
            and (
                not (
                    "nlp" in sig.parameters
                    and (
                        sig.parameters["nlp"].default is sig.empty
                        or sig.parameters["nlp"].annotation
                        in (Pipeline, PipelineProtocol)
                    )
                )
                or "nlp" in self._kwargs
            )
            and not self.search_nested_drafts(self._kwargs)
        ):
            return self._func(**self._kwargs)
        return self

    @classmethod
    def search_nested_drafts(cls, obj):
        if isinstance(obj, DraftPipe):
            return obj
        elif isinstance(obj, dict):
            for value in obj.values():
                result = cls.search_nested_drafts(value)
                if result is not None:
                    return result
        elif isinstance(obj, (tuple, list, set)):
            for value in obj:
                result = cls.search_nested_drafts(value)
                if result is not None:
                    return result
        return None

    def instantiate(
        self,
        nlp: "edsnlp.Pipeline",
        path: Optional[Sequence[str]] = (),
    ):
        """
        To ensure compatibility with spaCy's API, we need to support
        passing in the nlp object and name to factories. Since they can be
        nested, we need to add them to every factory in the config.
        """
        if isinstance(self, DraftPipe):
            if self.error is not None:
                raise self.error

            if self.instantiated is not None:
                return self.instantiated

            name = path[0] if len(path) == 1 else None
            parameters = (
                inspect.signature(self._func.__init__).parameters
                if isinstance(self._func, type)
                else inspect.signature(self._func).parameters
            )
            kwargs = {
                key: DraftPipe.instantiate(
                    self=value,
                    nlp=nlp,
                    path=(*path, key),
                )
                for key, value in self._kwargs.items()
            }
            try:
                if nlp and "nlp" in parameters:
                    kwargs["nlp"] = nlp
                if name and "name" in parameters:
                    kwargs["name"] = name
                self.instantiated = self._func(**kwargs)
            except ConfitValidationError as e:
                self.error = e
                raise ConfitValidationError(
                    patch_errors(e.raw_errors, path, model=e.model),
                    model=e.model,
                    name=self._func.__module__ + "." + self._func.__qualname__,
                )  # .with_traceback(None)
            # except Exception as e:
            #     obj.error = e
            #     raise ConfitValidationError([ErrorWrapper(e, path)])
            return self.instantiated
        elif isinstance(self, dict):
            instantiated = {}
            errors = []
            for key, value in self.items():
                try:
                    instantiated[key] = DraftPipe.instantiate(
                        self=value,
                        nlp=nlp,
                        path=(*path, key),
                    )
                except ConfitValidationError as e:
                    errors.extend(e.raw_errors)
            if errors:
                raise ConfitValidationError(errors)
            return instantiated
        elif isinstance(self, (tuple, list)):
            instantiated = []
            errors = []
            for i, value in enumerate(self):
                try:
                    instantiated.append(
                        DraftPipe.instantiate(value, nlp, (*path, str(i)))
                    )
                except ConfitValidationError as e:  # pragma: no cover
                    errors.append(e.raw_errors)
            if errors:
                raise ConfitValidationError(errors)
            return type(self)(instantiated)
        else:
            return self

    def _raise_draft_error(self):
        raise TypeError(
            f"This {self} component has not been instantiated "
            f"yet, likely because it was missing an `nlp` pipeline argument. You "
            f"should either:\n"
            f"- add it to a pipeline: `pipe = nlp.add_pipe(pipe)`\n"
            f"- or fill its `nlp` argument: `pipe = factory(nlp=nlp, ...)`"
        )


glob = []


def get_all_available_factories(namespaces):
    result = set()
    for ns in namespaces:
        entry_point_ns = "_".join(ns)
        eps = (
            catalogue.AVAILABLE_ENTRY_POINTS.select(group=entry_point_ns)
            if hasattr(catalogue.AVAILABLE_ENTRY_POINTS, "select")
            else catalogue.AVAILABLE_ENTRY_POINTS.get(entry_point_ns, [])
        )
        result.update([ep.name for ep in eps])
        for keys in catalogue.REGISTRY.copy().keys():
            if ns == keys[: len(ns)]:
                result.add(keys[-1])
    return sorted(result)


class FactoryRegistry(Registry):
    """
    A registry that validates the input arguments of the registered functions.
    """

    def get(self, name: str) -> Any:
        """
        Get the registered function for a given name.
        Since we want to be able to get functions registered under spacy namespace,
        and functions defined in memory but not accessible via entry points (spacy
        internal namespace), we need to check multiple registries.

        The strategy is the following:

        1. If the function exists in the edsnlp_factories namespace, return it as a
           curried function.
        2. Otherwise, check spacy namespaces and re-register it under edsnlp_factories
           namespace, then return it as a curried function.
        3. Otherwise, search in edsnlp's entry points and redo steps 1 & 2
        4. Otherwise, search in spacy's points and redo steps 1 & 2
        5. Fail

        name (str): The name.
        RETURNS (Any): The registered function.
        """

        registry_path = list(self.namespace) + [name]
        registry_spacy_path = ["spacy", "factories", name]
        registry_internal_spacy_path = ["spacy", "internal_factories", name]
        registry_legacy_spacy_path = ["spacy-legacy", "factories", name]

        def check_and_return():
            if not catalogue.check_exists(*registry_path):
                for path in (
                    registry_spacy_path,
                    registry_internal_spacy_path,
                    registry_legacy_spacy_path,
                ):
                    if catalogue.check_exists(*path):
                        func = catalogue._get(path)
                        meta = spacy.Language.get_factory_meta(name)

                        self.register(
                            name,
                            func=func,
                            assigns=meta.assigns,
                            requires=meta.requires,
                            retokenizes=meta.retokenizes,
                            default_config=meta.default_config,
                        )

            if catalogue.check_exists(*registry_path):
                func = catalogue._get(registry_path)
                return lambda **kwargs: DraftPipe(func, kwargs=kwargs).maybe_nlp()

        # Steps 1 & 2
        func = check_and_return()
        if func is None and self.entry_points:
            register_factories()
            # Update entry points in case packages lookup paths have changed
            catalogue.AVAILABLE_ENTRY_POINTS = importlib_metadata.entry_points()
            # Otherwise, step 3
            self.get_entry_point(name)
            # Then redo steps 1 & 2
            func = check_and_return()
        if func is None:
            if hasattr(spacy.registry, "_entry_point_factories"):
                spacy.registry._entry_point_factories.get_entry_point(name)
                # Then redo steps 1 & 2
                func = check_and_return()

        if func is not None:
            return func

        raise catalogue.RegistryError(
            (
                "Can't find '{name}' in registry {current_namespace}. "
                "Available names: {available_str}"
            ).format(
                name=name,
                current_namespace=" -> ".join(self.namespace),
                available_str=", ".join(
                    get_all_available_factories(
                        [
                            self.namespace,
                            ("spacy", "factories"),
                            ("spacy", "internal_factories"),
                        ]
                    )
                )
                or "none",
            )
        )

    def _get_entry_points(self) -> List[importlib_metadata.EntryPoint]:
        return (
            catalogue.AVAILABLE_ENTRY_POINTS.select(group=self.entry_point_namespace)
            if hasattr(catalogue.AVAILABLE_ENTRY_POINTS, "select")
            else catalogue.AVAILABLE_ENTRY_POINTS.get(self.entry_point_namespace, [])
        )

    def register(
        self,
        name: str,
        *,
        func: Optional[catalogue.InFunc] = None,
        default_config: Dict[str, Any] = FrozenDict(),
        assigns: Iterable[str] = FrozenList(),
        requires: Iterable[str] = FrozenList(),
        retokenizes: bool = False,
        default_score_weights: Dict[str, Optional[float]] = FrozenDict(),
        invoker: Callable = None,
        deprecated: Sequence[str] = (),
        spacy_compatible: bool = True,
    ) -> Callable[[catalogue.InFunc], catalogue.InFunc]:
        """
        This is a convenience wrapper around `confit.Registry.register`, that
        curries the function to be registered, allowing to instantiate the class
        later once `nlp` and `name` are known.

        Parameters
        ----------
        name: str
        func: Optional[catalogue.InFunc]
        default_config: Dict[str, Any]
        assigns: Iterable[str]
        requires: Iterable[str]
        retokenizes: bool
        default_score_weights: Dict[str, Optional[float]]
        invoker: Callable
        deprecated: Sequence[str]
        spacy_compatible: bool

        Returns
        -------
        Callable[[catalogue.InFunc], catalogue.InFunc]
        """
        save_params = {"@factory": name}

        def register(fn: catalogue.InFunc) -> catalogue.InFunc:
            assert (
                not spacy_compatible
                or len(accepted_arguments(fn, ["nlp", "name"])) == 2
            ), (
                "Spacy compatible factories functions must accept nlp and name as "
                "arguments. Either set register(..., spacy_compatible=False) or "
                "add nlp and name to the arguments."
            )

            meta = FactoryMeta(
                assigns=validate_attrs(assigns),
                requires=validate_attrs(requires),
                retokenizes=retokenizes,
                default_config=default_config,
            )

            def invoke(validated_fn, kwargs):
                if default_config is not None:
                    kwargs = (
                        Config(default_config)
                        .resolve(registry=self.registry)
                        .merge(kwargs)
                    )
                instantiated = (
                    invoker(validated_fn, kwargs)
                    if invoker is not None
                    else validated_fn(kwargs)
                )
                PIPE_META[instantiated] = meta
                return instantiated

            annotations = (
                fn.__annotations__
                if not isinstance(fn, type)
                else fn.__init__.__annotations__
            )
            old_annotation = None
            if "nlp" in annotations:  # annotations["nlp"] is spacy.Language:
                old_annotation = annotations["nlp"]
                annotations["nlp"] = Optional[edsnlp.core.PipelineProtocol]
            registered_fn = Registry.register(
                self,
                name=name,
                save_params=save_params,
                skip_save_params=["nlp", "name"],
                func=fn,
                invoker=invoke,
                deprecated=deprecated,
            )
            if old_annotation is not None:
                annotations["nlp"] = old_annotation

            if spacy_compatible:
                for spacy_pipe_name in (name, *deprecated):
                    spacy.Language.factory(
                        name=spacy_pipe_name,
                        default_config=default_config,
                        assigns=assigns,
                        requires=requires,
                        default_score_weights=default_score_weights,
                        retokenizes=retokenizes,
                        func=registered_fn,
                    )

            @wraps(fn)
            def curried_registered_fn(**kwargs):
                return DraftPipe(registered_fn, kwargs).maybe_nlp()

            return (
                curried_registered_fn
                if isinstance(fn, types.FunctionType)
                else registered_fn
            )

        return register(func) if func is not None else register


class registry(RegistryCollection):
    factories = FactoryRegistry(("edsnlp", "factories"), entry_points=True)
    factory = factories
    misc = Registry(("spacy", "misc"), entry_points=True)
    languages = Registry(("spacy", "languages"), entry_points=True)
    tokenizers = Registry(("spacy", "tokenizers"), entry_points=True)
    metrics = Registry(("spacy", "scorers"), entry_points=True)
    scorers = metrics
    accelerator = Registry(("edsnlp", "accelerator"), entry_points=True)
    adapters = Registry(("edsnlp", "adapters"), entry_points=True)
    readers = Registry(("edsnlp", "readers"), entry_points=True)
    writers = Registry(("edsnlp", "writers"), entry_points=True)
    core = Registry(("edsnlp", "core"), entry_points=True)
    optimizers = Registry(("edsnlp", "optimizers"), entry_points=True)
    schedules = Registry(("edsnlp", "schedules"), entry_points=True)
    loggers = Registry(("edsnlp", "loggers"), entry_points=True)


set_default_registry(registry)
