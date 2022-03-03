# Installation

You can install EDS-NLP via `pip`, like you would any other package.

## Simple installation

Installing EDS-NLP is straightforward :

<!-- termynal -->

```
$ pip install edsnlp
---> 100%
Installed
```

We recommend pinning the library version in your projects, or use a strict package manager like [Poetry](https://python-poetry.org/).

```
pip install edsnlp==0.4.0
```

## Development installation

To be able to run the test suite, run the example notebooks and develop your own pipeline, clone the repo and install it locally.

<!-- termynal -->

```
# Clone the repository and change directory
$ git clone https://github.com/aphp/edsnlp.git
---> 100%
$ cd edsnlp

# Optional: create a virtual environment
$ python -m venv venv
$ source venv/bin/activate

# Install setup dependencies and build resources
$ pip install -r requirements.txt
$ pip install -r requirements-setup.txt
$ python scripts/conjugate.py

# Install development dependencies
$ pip install -r requirements-dev.txt
$ pip install -r requirements-docs.txt

# Finally, install the package
$ pip install .
```

To make sure the pipeline will not fail because of formatting errors, we added pre-commit hooks using the `pre-commit` Python library. To use it, simply install it:

<!-- termynal -->

```
$ pre-commit install
```

The pre-commit hooks defined in the [configuration](https://gitlab.eds.aphp.fr/datasciencetools/edsnlp/-/blob/master/.pre-commit-config.yaml) will automatically run when you commit your changes, letting you know if something went wrong.

The hooks only run on staged changes. To force-run it on all files, run:

<!-- termynal -->

```
$ pre-commit run --all-files
---> 100%
All good !
```
