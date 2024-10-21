from confit import Cli

from edsnlp.core.registries import registry
from edsnlp.training import train

app = Cli(pretty_exceptions_show_locals=False)
train_command = app.command(name="train", registry=registry)(train)

if __name__ == "__main__":
    app()
