from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union

import pydantic
import regex
import torch
import torch.optim
from confit import Config, validate_arguments
from typing_extensions import Literal

import edsnlp
from edsnlp.core import PipelineProtocol
from edsnlp.utils.collections import get_deep_attr, set_deep_attr
from edsnlp.utils.typing import AsList

optim_mapping = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adamax": torch.optim.Adamax,
    "rmsprop": torch.optim.RMSprop,
}


@validate_arguments
class Schedule:
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
        path: str = "lr",
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
        self.path = path
        self.start_value = start_value
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
            self.max_value = get_deep_attr(group, self.path)
        warmup_steps = self.total_steps * self.warmup_rate
        if self.idx < warmup_steps:
            progress = self.idx / warmup_steps
            value = self.start_value + (self.max_value - self.start_value) * progress
        else:
            progress = (self.idx - warmup_steps) / (self.total_steps - warmup_steps)
            value = self.max_value + (0 - self.max_value) * progress
        set_deep_attr(group, self.path, value)

    def __repr__(self):
        format_string = type(self).__name__ + "(\n"
        format_string += f"    start_value: {self.start_value}\n"
        format_string += f"    max_value: {self.max_value}\n"
        format_string += f"    warmup_rate: {self.warmup_rate}\n"
        format_string += f"    path: {self.path}\n"
        format_string += f"    total_steps: {self.total_steps}\n"
        format_string += ")"
        return format_string


class Group(pydantic.BaseModel, extra=pydantic.Extra.allow):
    """
    Parameter group for the optimizer.

    Parameters
    ----------
    schedules : AsList[Schedule]
        The schedules to apply to the group.
    lr : Optional[float] = None
        The learning rate for the group.
    **kwargs
        Additional parameters to pass to the group.
    """

    schedules: Optional[AsList[Schedule]] = None
    lr: Optional[float] = None


if TYPE_CHECKING:
    Group = Dict


@edsnlp.registry.misc.register("eds.scheduled_optimizer")
class ScheduledOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        cls: Union[torch.optim.Optimizer, Type[torch.optim.Optimizer], str],
        module: Optional[Union[PipelineProtocol, torch.nn.Module]] = None,
        total_steps: Optional[int] = None,
        groups: Optional[Dict[str, Union[Group, Literal[False]]]] = None,
        init_schedules: bool = True,
        **kwargs,
    ):
        """
        Wrapper optimizer that supports schedules for the parameters and easy parameter
        selection using the key of the `groups` dictionary as regex patterns to match
        the parameter names.

        Parameters
        ----------
        cls : Union[str, Type[torch.optim.Optimizer], torch.optim.Optimizer]
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
            matter. A parameter will be assigned to the first group that matches it, so
            you can also exclude parameters by using a selector early in the groups and
            putting `False` as the value.
        """
        should_instantiate_optim = isinstance(cls, str) or isinstance(cls, type)
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
            all_matched_params = set()
            for sel, group in groups.items():
                params = []
                for name, param in named_parameters:
                    if param not in all_matched_params and regex.search(sel, name):
                        params.append(param)
                if group:
                    tmp_group = dict(group)
                    group.clear()
                    group: Dict
                    group["selector"] = sel
                    group["params"] = params
                    group.update(tmp_group)
                all_matched_params |= set(params)
            groups = [
                {k: v for k, v in group.items() if v is not None}
                for group in groups.values()
                if group
            ]

            if isinstance(cls, str):
                cls = (
                    optim_mapping[cls.lower()]
                    if cls.lower() in optim_mapping
                    else getattr(torch.optim, cls)
                )
            cls = cls(groups, **kwargs)

        self.optim = cls
        schedule_to_groups = defaultdict(lambda: [])
        for group in self.optim.param_groups:
            if "schedules" in group:
                group["schedules"] = (
                    group["schedules"]
                    if isinstance(group["schedules"], list)
                    else [group["schedules"]]
                )
                group["schedules"] = list(group["schedules"])
                for schedule in group["schedules"]:
                    schedule_to_groups[schedule].append(group)
                    if schedule.total_steps is None:
                        assert total_steps is not None, (
                            "total_steps must be provided to the optimizer "
                            "or the schedule"
                        )
                        schedule.total_steps = total_steps
                    if init_schedules:
                        schedule.step(group)

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
            "lr": [group.get("lr") for group in self.optim.param_groups],
            "schedules": [
                [schedule.state_dict() for schedule in group.get("schedules", ())]
                for group in self.optim.param_groups
            ],
        }
        for group in state["optim"]["param_groups"]:
            if "schedules" in group:
                del group["schedules"]
        return state

    def load_state_dict(self, state):
        optim_schedules = [
            group.get("schedules", ()) for group in self.optim.param_groups
        ]
        self.optim.load_state_dict(state["optim"])
        for group, group_schedule, group_schedules_state, lr in zip(
            self.optim.param_groups, optim_schedules, state["schedules"], state["lr"]
        ):
            group["schedules"] = group_schedule
            for schedule, schedule_state in zip(
                group["schedules"], group_schedules_state
            ):
                schedule.load_state_dict(schedule_state)
            group["lr"] = lr

    def step(self, closure=None):
        self.optim.step(closure=closure)
        for group in self.optim.param_groups:
            if "schedules" in group:
                for schedule in group["schedules"]:
                    schedule.step(group)

    def initialize(self):
        self.reset()
        self.optim.step()

    def reset(self):
        self.optim.zero_grad()
        for group in self.optim.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    param.grad = torch.zeros_like(param)
        for group in self.optim.param_groups:
            if "schedules" in group:
                for schedule in group["schedules"]:
                    schedule.reset(group)

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
                *sorted(set(group.keys()) - {"selector", "params", "lr", "schedules"}),
                "schedules",
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
