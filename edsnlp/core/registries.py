import inspect
import types
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union
from weakref import WeakKeyDictionary

import catalogue
import spacy
from confit import Config, Registry, RegistryCollection, set_default_registry
from confit.errors import ConfitValidationError, patch_errors
from spacy.pipe_analysis import validate_attrs

import edsnlp
from edsnlp.utils.collections import FrozenDict, FrozenList

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

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


class CurriedFactory:
    def __init__(self, func, kwargs):
        self.kwargs = kwargs
        self.factory = func
        self.instantiated = None
        self.error = None

    def maybe_nlp(self) -> Union["CurriedFactory", Any]:
        """
        If the factory requires an nlp argument and the user has explicitly
        provided it (this is unusual, we usually expect the factory to be
        instantiated via add_pipe, or a config), then we should instantiate
        it.

        Returns
        -------
        Union["CurriedFactory", Any]
        """
        from edsnlp.core.pipeline import Pipeline, PipelineProtocol

        sig = inspect.signature(self.factory)
        if (
            not (
                "nlp" in sig.parameters
                and (
                    sig.parameters["nlp"].default is sig.empty
                    or sig.parameters["nlp"].annotation in (Pipeline, PipelineProtocol)
                )
            )
            or "nlp" in self.kwargs
        ) and not self.search_curried_factory(self.kwargs):
            return self.factory(**self.kwargs)
        return self

    @classmethod
    def search_curried_factory(cls, obj):
        if isinstance(obj, CurriedFactory):
            return obj
        elif isinstance(obj, dict):
            for value in obj.values():
                result = cls.search_curried_factory(value)
                if result is not None:
                    return result
        elif isinstance(obj, (tuple, list, set)):
            for value in obj:
                result = cls.search_curried_factory(value)
                if result is not None:
                    return result
        return None

    def instantiate(
        obj: Any,
        nlp: "edsnlp.Pipeline",
        path: Optional[Sequence[str]] = (),
    ):
        """
        To ensure compatibility with spaCy's API, we need to support
        passing in the nlp object and name to factories. Since they can be
        nested, we need to add them to every factory in the config.
        """
        if isinstance(obj, CurriedFactory):
            if obj.error is not None:
                raise obj.error

            if obj.instantiated is not None:
                return obj.instantiated

            name = path[0] if len(path) == 1 else None
            parameters = (
                inspect.signature(obj.factory.__init__).parameters
                if isinstance(obj.factory, type)
                else inspect.signature(obj.factory).parameters
            )
            kwargs = {
                key: CurriedFactory.instantiate(
                    obj=value,
                    nlp=nlp,
                    path=(*path, key),
                )
                for key, value in obj.kwargs.items()
            }
            try:
                if nlp and "nlp" in parameters:
                    kwargs["nlp"] = nlp
                if name and "name" in parameters:
                    kwargs["name"] = name
                obj.instantiated = obj.factory(**kwargs)
            except ConfitValidationError as e:
                obj.error = e
                raise ConfitValidationError(
                    patch_errors(e.raw_errors, path, model=e.model),
                    model=e.model,
                    name=obj.factory.__module__ + "." + obj.factory.__qualname__,
                )  # .with_traceback(None)
            # except Exception as e:
            #     obj.error = e
            #     raise ConfitValidationError([ErrorWrapper(e, path)])
            return obj.instantiated
        elif isinstance(obj, dict):
            instantiated = {}
            errors = []
            for key, value in obj.items():
                try:
                    instantiated[key] = CurriedFactory.instantiate(
                        obj=value,
                        nlp=nlp,
                        path=(*path, key),
                    )
                except ConfitValidationError as e:
                    errors.extend(e.raw_errors)
            if errors:
                raise ConfitValidationError(errors)
            return instantiated
        elif isinstance(obj, (tuple, list)):
            instantiated = []
            errors = []
            for i, value in enumerate(obj):
                try:
                    instantiated.append(
                        CurriedFactory.instantiate(value, nlp, (*path, str(i)))
                    )
                except ConfitValidationError as e:  # pragma: no cover
                    errors.append(e.raw_errors)
            if errors:
                raise ConfitValidationError(errors)
            return type(obj)(instantiated)
        else:
            return obj

    def _raise_curried_factory_error(self):
        raise TypeError(
            f"This component CurriedFactory({self.factory}) has not been instantiated "
            f"yet, likely because it was missing an `nlp` pipeline argument. You "
            f"should either:\n"
            f"- add it to a pipeline: `pipe = nlp.add_pipe(pipe)`\n"
            f"- or fill its `nlp` argument: `pipe = factory(nlp=nlp, ...)`"
        )

    def __call__(self, *args, **kwargs):
        self._raise_curried_factory_error()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        self._raise_curried_factory_error()

    def __repr__(self):
        return f"CurriedFactory({self.factory})"


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

        def check_and_return():
            if not catalogue.check_exists(*registry_path):
                for path in (registry_spacy_path, registry_internal_spacy_path):
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
                return lambda **kwargs: CurriedFactory(func, kwargs=kwargs).maybe_nlp()

        # Steps 1 & 2
        func = check_and_return()
        if func is None and self.entry_points:
            # Update entry points in case packages lookup paths have changed
            catalogue.AVAILABLE_ENTRY_POINTS = importlib_metadata.entry_points()
            # Otherwise, step 3
            self.get_entry_point(name)
            # Then redo steps 1 & 2
            func = check_and_return()
        if func is None:
            # Otherwise, step 4
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
                return CurriedFactory(registered_fn, kwargs).maybe_nlp()

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


set_default_registry(registry)
