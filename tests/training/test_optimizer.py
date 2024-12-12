# ruff:noqa:E402
import pytest

try:
    import torch.nn
except ImportError:
    torch = None

if torch is None:
    pytest.skip("torch not installed", allow_module_level=True)
pytest.importorskip("rich")

from edsnlp.training.optimizer import LinearSchedule, ScheduledOptimizer


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(10, 1)
        self.fc2 = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture(scope="module")
def net():
    net = Net()
    return net


@pytest.mark.parametrize(
    "groups",
    [
        # Old schedule API
        {
            "fc1[.].*": {
                "lr": 0.1,
                "weight_decay": 0.01,
                "schedules": [
                    {
                        "@schedules": "linear",
                        "start_value": 0.0,
                        "warmup_rate": 0.2,
                    },
                ],
            },
            "fc2[.]bias": False,
            "": {
                "lr": 0.0001,
                "weight_decay": 0.0,
            },
        },
        # New schedule API
        {
            "fc1[.].*": {
                "lr": {
                    "@schedules": "linear",
                    "start_value": 0.0,
                    "max_value": 0.1,
                    "warmup_rate": 0.2,
                },
                "weight_decay": 0.01,
            },
            "fc2[.]bias": False,
            "": {
                "lr": 0.0001,
                "weight_decay": 0.0,
            },
        },
    ],
)
def test_old_parameter_selection(net, groups):
    optim = ScheduledOptimizer(
        optim="adamw",
        module=net,
        groups=groups,
        total_steps=10,
    )
    assert len(optim.state) == 0
    optim.initialize()
    assert all([p in optim.state for p in net.fc1.parameters()])
    optim.state = optim.state

    fc1_group = optim.param_groups[1]
    assert fc1_group["lr"] == pytest.approx(0.0)
    assert fc1_group["weight_decay"] == pytest.approx(0.01)
    assert set(fc1_group["params"]) == {net.fc1.weight, net.fc1.bias}

    fc2_group = optim.param_groups[0]
    assert fc2_group["lr"] == pytest.approx(0.0001)
    assert set(fc2_group["params"]) == {net.fc2.weight}

    lr_values = [fc1_group["lr"]]

    for i in range(10):
        optim.step()
        lr_values.append(fc1_group["lr"])

    assert lr_values == pytest.approx(
        [
            0.0,
            0.05,
            0.1,
            0.0875,
            0.075,
            0.0625,
            0.05,
            0.0375,
            0.025,
            0.0125,
            0.0,
        ]
    )


def test_serialization(net):
    optim = ScheduledOptimizer(
        optim="adamw",
        module=net,
        groups={
            "fc1[.].*": {
                "lr": 0.1,
                "weight_decay": 0.01,
                "schedules": LinearSchedule(start_value=0.0, warmup_rate=0.2),
            },
            "fc2[.]bias": False,
            "": {
                "lr": 0.0001,
                "weight_decay": 0.0,
            },
        },
        total_steps=10,
    )
    optim.initialize()
    optim.param_groups = optim.param_groups

    state_dict = None
    for i in range(10):
        if i == 5:
            state_dict = optim.state_dict()
        optim.step()

    assert optim.param_groups[-1]["lr"] == pytest.approx(0.0)
    optim.load_state_dict(state_dict)
    assert optim.param_groups[-1]["lr"] == pytest.approx(0.0625)

    optim.reset()


def test_repr(net):
    optim = ScheduledOptimizer(
        optim="adamw",
        module=net,
        groups={
            "fc1[.].*": {
                "lr": 0.1,
                "weight_decay": 0.01,
                "schedules": [
                    LinearSchedule(start_value=0.0, warmup_rate=0.2),
                    LinearSchedule(path="weight_decay"),
                ],
            },
            "fc2[.]bias": False,
            ".*": {
                "lr": 0.0001,
                "weight_decay": 0.0,
            },
        },
        total_steps=10,
    )
    optim.initialize()

    assert "ScheduledOptimizer[AdamW]" in repr(optim)
