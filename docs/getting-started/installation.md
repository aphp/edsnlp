# Installation

## Simple installation

Installing EDS-NLP is straightforward :

```shell
pip install git+https://gitlab.eds.aphp.fr/equipedatascience/edsnlp.git
```

EDS-NLP is still young, and subject to rapid (as well as backward-incompatible) changes. We thus recommend pinning the library version in your projects :

```shell
pip install git+https://gitlab.eds.aphp.fr/equipedatascience/edsnlp.git@v0.3.2
```

## Development installation

To be able to run the test suite, run the example notebooks and develop your own pipeline, clone the repo and install it locally :

```shell
git clone https://gitlab.eds.aphp.fr/equipedatascience/edsnlp.git
cd edsnlp
pip install .
```

You should also install development librairies with:

```shell
pip install -r requirements-dev.txt

# If you plan on building the documentation locally
pip install -r requirements-doc.txt
```

To make sure the pipeline will not fail because of formatting errors, we added pre-commit hooks using the `pre-commit` Python library. To use it, simply install it:

```shell
pre-commit install
```

The pre-commit hooks defined in the [configuration](https://gitlab.eds.aphp.fr/datasciencetools/edsnlp/-/blob/master/.pre-commit-config.yaml) will automatically run when you commit your changes, letting you know if something went wrong.

The hooks only run on staged changes. To force-run it on all files, run:

```shell
pre-commit run --all-files
```
