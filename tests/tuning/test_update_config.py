import pytest

try:
    import optuna
except ImportError:
    optuna = None

if optuna is None:
    pytest.skip("optuna not installed", allow_module_level=True)


from edsnlp.tune import update_config


@pytest.fixture
def minimal_config():
    return {
        "train": {
            "layers": None,
        },
        ".lr": {
            "learning_rate": None,
        },
    }


@pytest.fixture
def hyperparameters():
    return {
        "train.layers": {
            "type": "int",
            "low": 2,
            "high": 8,
            "step": 2,
        },
        "'.lr'.learning_rate": {
            "alias": "learning_rate",
            "type": "float",
            "low": 0.001,
            "high": 0.1,
            "log": True,
        },
        "train.batch_size": {
            "alias": "batch_size",
            "type": "categorical",
            "choices": [32, 64, 128],
        },
    }


@pytest.fixture
def hyperparameters_with_invalid_type():
    return {
        "train.optimizer": {
            "type": "string",
            "choices": ["adam", "sgd"],
        }
    }


@pytest.fixture
def hyperparameters_with_invalid_path():
    return {
        "model.layers": {
            "type": "int",
            "low": 2,
            "high": 8,
            "step": 2,
        },
    }


@pytest.fixture
def trial():
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    return trial


def test_update_config_with_values(minimal_config, hyperparameters):
    values = {"learning_rate": 0.05, "train.layers": 6, "batch_size": 64}
    _, updated_config = update_config(minimal_config, hyperparameters, values=values)

    assert updated_config[".lr"]["learning_rate"] == values["learning_rate"]
    assert updated_config["train"]["layers"] == values["train.layers"]
    assert updated_config["train"]["batch_size"] == values["batch_size"]


def test_update_config_with_trial(minimal_config, hyperparameters, trial):
    _, updated_config = update_config(minimal_config, hyperparameters, trial=trial)

    learning_rate = updated_config[".lr"]["learning_rate"]
    layers = updated_config["train"]["layers"]
    batch_size = updated_config["train"]["batch_size"]

    assert (
        hyperparameters["'.lr'.learning_rate"]["low"]
        <= learning_rate
        <= hyperparameters["'.lr'.learning_rate"]["high"]
    )
    assert (
        hyperparameters["train.layers"]["low"]
        <= layers
        <= hyperparameters["train.layers"]["high"]
    )
    assert layers % hyperparameters["train.layers"]["step"] == 0
    assert batch_size in hyperparameters["train.batch_size"]["choices"]


def test_update_config_raises_error_on_unknown_parameter_type(
    minimal_config, hyperparameters_with_invalid_type, trial
):
    with pytest.raises(
        ValueError,
        match="Unknown parameter type 'string' for hyperparameter 'train.optimizer'.",
    ):
        update_config(minimal_config, hyperparameters_with_invalid_type, trial=trial)


def test_update_config_raises_error_on_wrong_path(
    minimal_config, hyperparameters_with_invalid_path, trial
):
    with pytest.raises(KeyError, match="Path 'model' not found in config."):
        update_config(minimal_config, hyperparameters_with_invalid_path, trial=trial)
