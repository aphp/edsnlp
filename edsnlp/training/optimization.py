import fnmatch
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
from confit import Config

import edsnlp
from edsnlp.utils.collections import get_deep_attr, set_deep_attr


class ScheduledOptimizer(torch.optim.Optimizer):
    def __init__(self, optim, init_schedules: bool = True):
        self.optim = optim
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


class OptimizerGroupsProxy:
    def __init__(self, groups):
        self.param_groups = groups


@edsnlp.registry.schedules.register("linear")
class LinearSchedule:
    def __init__(
        self,
        total_steps: Optional[int] = None,
        max_value: Optional[Any] = None,
        start_value: float = 0.0,
        path: str = "lr",
        warmup_rate: float = 0.0,
    ):
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
        return (
            f"LinearSchedule(total_steps={self.total_steps}, "
            f"max_value={self.max_value}, "
            f"start_value={self.start_value}, "
            f"path={self.path}, "
            f"warmup_rate={self.warmup_rate})"
        )


def create_optimizer(optim, **kwargs):
    def instantiate(nlp, total_steps=None):
        groups = list(nlp.parameters())
        named_parameters = list(nlp.named_parameters())
        if "groups" in kwargs:
            optim_groups = Config.resolve(
                kwargs.pop("groups"), registry=edsnlp.registry
            )
            optim_groups = {
                sel: dict(group) if group else False
                for sel, group in optim_groups.items()
            }
            all_matched_params = set()
            for sel, group in optim_groups.items():
                params = []
                for name, param in named_parameters:
                    if param not in all_matched_params and fnmatch.fnmatch(name, sel):
                        params.append(param)
                if group:
                    group: Dict
                    group["selector"] = sel
                    group["params"] = params
                all_matched_params |= set(params)
            groups = [group for group in optim_groups.values() if group]
        instance = ScheduledOptimizer(
            optim(groups, **kwargs),
            init_schedules=False,
        )
        for group in instance.param_groups:
            if "schedules" in group:
                for schedule in group["schedules"]:
                    if schedule.total_steps is None:
                        assert total_steps is not None, (
                            "total_steps must be provided to the optimizer "
                            "or the schedule"
                        )
                        schedule.total_steps = total_steps
                    schedule.step(group)
        return instance

    return (
        instantiate
        if "nlp" not in kwargs
        else instantiate(kwargs.pop("nlp"), kwargs.pop("total_steps", None))
    )


for optim_name, optim in vars(torch.optim).items():
    if (
        isinstance(optim, type)
        and issubclass(optim, torch.optim.Optimizer)
        and optim is not torch.optim.Optimizer
    ):

        def wrapper(optim):
            def create_specific_optimizer(**kwargs):
                return create_optimizer(optim=optim, **kwargs)

            return create_specific_optimizer

        edsnlp.registry.optimizers.register(optim_name.lower(), func=wrapper(optim))
