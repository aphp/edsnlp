import importlib
from collections import defaultdict
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import regex
import torch
import torch.optim
from confit import Config, validate_arguments
from confit.utils.collections import split_path
from typing_extensions import Literal

import edsnlp
from edsnlp.core import PipelineProtocol
from edsnlp.utils.collections import get_deep_attr, set_deep_attr

optim_mapping = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adamax": torch.optim.Adamax,
    "rmsprop": torch.optim.RMSprop,
}


def get_optimizer(optim):
    if isinstance(optim, str):
        optim_lower = optim.lower()
        if optim_lower in optim_mapping:
            return optim_mapping[optim_lower]
        else:
            try:
                # Attempt to get the optimizer from torch.optim
                return getattr(torch.optim, optim)
            except AttributeError:
                # If not found in torch.optim, try to import it dynamically
                module_name, class_name = optim.rsplit(".", 1)
                module = importlib.import_module(module_name)
                return getattr(module, class_name)
    return optim


@validate_arguments
class Schedule:
    paths: List[Tuple[Union[str, int]]]
    start_value: Any

    def __init__(self, path, start_value: Any):
        self.paths: Optional[List[Tuple[Union[str, int]]]] = (
            None
            if path is None
            else [(path if isinstance(path, list) else split_path(path))]
        )
        self.start_value = start_value

    def step(self, group, closure=None):
        raise NotImplementedError

    def reset(self, group):
        raise NotImplementedError()

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self, state):
        raise NotImplementedError()


@edsnlp.registry.schedules.register("linear")
class LinearSchedule(Schedule):
    def __init__(
        self,
        total_steps: Optional[int] = None,
        max_value: Optional[Any] = None,
        start_value: float = 0.0,
        path: Optional[Union[str, int, List[Union[str, int]]]] = None,
        warmup_rate: float = 0.0,
    ):
        """
        Linear schedule for a parameter group. The schedule will linearly increase
        the value from `start_value` to `max_value` in the first `warmup_rate` of the
        `total_steps` and then linearly decrease it to `0`.

        Parameters
        ----------
        total_steps: Optional[int]
            The total number of steps, usually used to calculate ratios.
        max_value: Optional[Any]
            The maximum value to reach.
        start_value: float
            The initial value.
        path: str
            The path to the attribute to set.
        warmup_rate: float
            The rate of the warmup.
        """
        super().__init__(path, start_value)
        self.max_value = max_value
        self.warmup_rate = warmup_rate
        self.total_steps = total_steps
        self.idx = 0

    def reset(self, group):
        self.idx = -1
        self.step(group)

    def state_dict(self):
        return {
            "idx": self.idx,
        }

    def load_state_dict(self, state):
        self.idx = state["idx"]

    def step(self, group, closure=None):
        self.idx += 1
        if self.max_value is None:
            self.max_value = get_deep_attr(group, self.paths[0])
        warmup_steps = self.total_steps * self.warmup_rate
        if self.idx < warmup_steps:
            progress = self.idx / warmup_steps
            value = self.start_value + (self.max_value - self.start_value) * progress
        else:
            progress = min(
                1.0, (self.idx - warmup_steps) / (self.total_steps - warmup_steps)
            )
            value = self.max_value + (0 - self.max_value) * progress
        for path in self.paths:
            set_deep_attr(group, path, value)

    def __repr__(self):
        format_string = type(self).__name__ + "(\n"
        format_string += f"    start_value: {self.start_value}\n"
        format_string += f"    max_value: {self.max_value}\n"
        format_string += f"    warmup_rate: {self.warmup_rate}\n"
        format_string += f"    paths: {self.paths}\n"
        format_string += f"    total_steps: {self.total_steps}\n"
        format_string += ")"
        return format_string


@edsnlp.registry.core.register("optimizer")
class ScheduledOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        optim: Union[torch.optim.Optimizer, Type[torch.optim.Optimizer], str],
        module: Optional[Union[PipelineProtocol, torch.nn.Module]] = None,
        total_steps: Optional[int] = None,
        groups: Optional[Dict[str, Union[Dict, Literal[False]]]] = None,
        init_schedules: bool = True,
        **kwargs,
    ):
        """
        Wrapper optimizer that supports schedules for the parameters and easy parameter
        selection using the key of the `groups` dictionary as regex patterns to match
        the parameter names.

        Schedules are defined directly in the groups, in place of the scheduled value.

        Examples
        --------
        ```{ .python .no-check }
        optim = ScheduledOptimizer(
            cls="adamw",
            module=model,
            groups={
                # Exclude all parameters matching 'bias' from optimization.
                "bias": False,
                # Parameters starting with 'transformer' receive this learning rate
                # schedule. If a parameter matches both 'transformer' and 'ner',
                # the 'transformer' settings take precedence due to the order.
                "^transformer": {
                    "lr": {
                        "@schedules": "linear",
                        "start_value": 0.0,
                        "max_value": 5e-4,
                        "warmup_rate": 0.2,
                    },
                },
                # Parameters starting with 'ner' receive this learning rate schedule,
                # unless a 'lr' value has already been set by an earlier selector.
                "^ner": {
                    "lr": {
                        "@schedules": "linear",
                        "start_value": 0.0,
                        "max_value": 1e-4,
                        "warmup_rate": 0.2,
                    },
                },
                # Apply a weight_decay of 0.01 to all parameters not excluded.
                # This setting doesn't conflict with others and applies to all.
                "": {
                    "weight_decay": 0.01,
                },
            },
            total_steps=1000,
        )
        ```

        Parameters
        ----------
        optim : Union[str, Type[torch.optim.Optimizer], torch.optim.Optimizer]
            The optimizer to use. If a string (like "adamw") or a type to instantiate,
            the`module` and `groups` must be provided.
        module : Optional[Union[PipelineProtocol, torch.nn.Module]]
            The module to optimize. Usually the `nlp` pipeline object.
        total_steps : Optional[int]
            The total number of steps, used for schedules.
        groups : Optional[Dict[str, Group]]
            The groups to optimize. The key is a regex selector to match parameters in
            `module.named_parameters()` and the value is a dictionary with the keys
            `params` and `schedules`.

            The matching is performed by running  `regex.search(selector, name)` so you
            do not have to match the full name. Note that the order of dict keys
            matter. If a parameter name matches multiple selectors, the
            configurations of these selectors are combined in reverse order (from the
            last matched selector to the first), allowing later selectors to complete
            options from earlier ones. If a selector maps to `False`, any parameters
            matching it are excluded from optimization and not included in any parameter
            group.
        """
        should_instantiate_optim = isinstance(optim, str) or isinstance(optim, type)
        if should_instantiate_optim and (groups is None or module is None):
            raise ValueError(
                "If the optimizer is a string or a type, the module and groups must "
                "be provided."
            )
        elif not should_instantiate_optim and (
            groups is not None or module is not None
        ):
            raise ValueError(
                "If the optimizer is already instantiated, the module and groups must "
                "not be provided."
            )

        if should_instantiate_optim:
            named_parameters = list(module.named_parameters())
            groups = Config.resolve(groups, registry=edsnlp.registry)
            groups = {
                sel: dict(group) if group else False for sel, group in groups.items()
            }
            param_to_groups = {}
            for name, param in named_parameters:
                param_to_groups[param] = tuple(
                    dict.fromkeys(
                        sel for sel, group in groups.items() if regex.search(sel, name)
                    )
                )
            groups_to_params = defaultdict(lambda: [])
            for params, group in param_to_groups.items():
                groups_to_params[group].append(params)

            cliques = []
            for selectors, params in groups_to_params.items():
                group = {}
                group_sources = {}
                for sel in reversed(selectors):
                    if groups[sel] is False:
                        break
                    group.update(groups[sel])
                    group_sources.update({k: sel for k in groups[sel]})
                else:
                    # if no group=False (break) was encountered
                    if group and "lr" in group and params:
                        sources = [
                            sel for sel in selectors if sel in group_sources.values()
                        ]
                        group["selectors"] = sources
                        group["params"] = params
                        cliques.append(group)
            cliques = reversed(
                [{k: v for k, v in group.items() if v is not None} for group in cliques]
            )

            optim = get_optimizer(optim)
            optim = optim(cliques, **kwargs)

        self.optim = optim
        self.schedules = self.extract_schedules(optim.param_groups)
        for schedule in self.schedules:
            if schedule.total_steps is None:
                assert (
                    total_steps is not None
                ), "total_steps must be provided to the optimizer or the schedule"
                schedule.total_steps = total_steps
            if init_schedules:
                schedule.step(optim.param_groups)

    @classmethod
    def extract_schedules(cls, param_groups):
        schedules = defaultdict(set)

        def rec(node, path):
            if len(path) == 2 and path[1] in ("schedules", "params"):
                return
            if isinstance(node, dict):
                items = node.items()
            elif isinstance(node, (list, tuple)):
                items = enumerate(node)
            else:
                if isinstance(node, Schedule):
                    schedules[node].add(path)
                return
            for key, value in items:
                rec(value, (*path, key))

        # For backward compatibility when schedules were defined
        # under the "schedules" key in groups.
        for i, group in enumerate(param_groups):
            if "schedules" in group:
                grp_schedules = group["schedules"]
                grp_schedules = (
                    grp_schedules
                    if isinstance(grp_schedules, list)
                    else [grp_schedules]
                )
                for schedule in grp_schedules:
                    if schedule.paths is None:
                        schedule.paths = [("lr",)]
                    schedule.paths = [(i, *path) for path in schedule.paths]
                    schedules[schedule].update(schedule.paths)

        rec(param_groups, ())

        for schedule, paths in schedules.items():
            paths = sorted(paths)
            if schedule.paths is None:
                schedule.paths = paths
            if schedule.paths != paths:
                raise ValueError(f"Schedule path mismatch: {schedule.paths} != {paths}")

        return list(schedules.keys())

    def zero_grad(self):
        return self.optim.zero_grad()

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    @property
    def state(self):
        return self.optim.state

    @state.setter
    def state(self, value):
        self.optim.state = value

    def state_dict(self):
        state = {
            "optim": self.optim.state_dict(),
            "schedules": [schedule.state_dict() for schedule in self.schedules],
        }
        return state

    def load_state_dict(self, state):
        self.optim.load_state_dict(state["optim"])
        for schedule_state, schedule in zip(state["schedules"], self.schedules):
            schedule.load_state_dict(schedule_state)

    def step(self, closure=None):
        self.optim.step(closure=closure)
        self.step_schedules()

    def step_schedules(self):
        for schedule in self.schedules:
            schedule.step(self.param_groups)

    def initialize(self):
        self.reset()
        self.optim.step()

    def reset(self):
        self.optim.zero_grad()
        for group in self.optim.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    param.grad = torch.zeros_like(param)
        for schedule in self.schedules:
            schedule.reset(self.param_groups)

    def __repr__(self):
        format_string = type(self).__name__ + f"[{type(self.optim).__qualname__}] ("
        ind = "    "
        for i, group in enumerate(self.param_groups):
            format_string += "\n"
            format_string += f"Parameter Group {i}\n"
            keys = [
                "selector",
                "params",
                "lr",
                *sorted(set(group.keys()) - {"selector", "params", "lr"}),
            ]
            for key in keys:
                if key in group:
                    format_string += ind + f"{key}: "
                    if key == "params":
                        num_tensors = len(group["params"])
                        num_params = sum(p.numel() for p in group["params"])
                        format_string += (
                            f"{num_params} parameters ({num_tensors} tensors)\n"
                        )
                    elif isinstance(group[key], list):
                        format_string += "[\n"
                        for i, item in enumerate(group[key]):
                            sub_str = str(item)
                            for line in sub_str.split("\n"):
                                format_string += ind * 2 + line + "\n"
                        format_string += ind + "]\n"
                    else:
                        sub_str = str(group[key])
                        for i, line in enumerate(sub_str.split("\n")):
                            format_string += (ind if i > 0 else "") + line + "\n"

        format_string += ")"
        return format_string
