from confit import Cli

from edsnlp.training.trainer import *  # noqa: F403
from edsnlp.training.trainer import registry, train

app = Cli(pretty_exceptions_show_locals=False)
train_command = app.command(name="train", registry=registry)(train)

if __name__ == "__main__":
    app()
