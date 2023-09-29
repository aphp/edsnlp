from collections import defaultdict

import torch

from edsnlp.utils.collections import get_deep_attr, set_deep_attr


class ScheduledOptimizer(torch.optim.Optimizer):
    def __init__(self, optim):
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


class OptimizerGroupsProxy:
    def __init__(self, groups):
        self.param_groups = groups


class LinearSchedule:
    def __init__(
        self,
        total_steps,
        max_value=None,
        start_value=0.0,
        path="lr",
        warmup_rate=0.0,
    ):
        self.path = path
        self.start_value = start_value
        self.max_value = max_value
        self.warmup_rate = warmup_rate
        self.total_steps = total_steps
        self.idx = 0

    def state_dict(self):
        return {
            "idx": self.idx,
        }

    def load_state_dict(self, state):
        self.idx = state["idx"]

    def step(self, group, closure=None):
        if self.max_value is None:
            self.max_value = get_deep_attr(group, self.path)
        warmup_steps = self.total_steps * self.warmup_rate
        if self.idx < warmup_steps:
            progress = self.idx / warmup_steps
            value = self.start_value + (self.max_value - self.start_value) * progress
        else:
            progress = (self.idx - warmup_steps) / (self.total_steps - warmup_steps)
            value = self.max_value + (0 - self.max_value) * progress
        self.idx += 1
        set_deep_attr(group, self.path, value)
