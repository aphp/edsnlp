# Contributing to EDS-NLP

We welcome contributions ! There are many ways to help. For example, you can:

1. Help us track bugs by filing issues
2. Suggest and help prioritise new functionalities
3. Develop a new pipeline ! Fork the project and propose a new functionality through a pull request
4. Help us make the library as straightforward as possible, by simply asking questions on whatever does not seem clear to you.

## Proposing a merge request

At the very least, your changes should :

- Be well-documented ;
- Pass every tests, and preferably implement its own ;
- Follow the style guide.

### Testing your code

We use the Pytest test suite.

The following command will run the test suite. Writing your own tests is encouraged !

```shell
python -m pytest
```

Should your contribution propose a bug fix, we require the bug be thoroughly tested.

### Architecture of a pipeline

All pipelines should follow the same pattern :

```
edsnlp/pipelines/<pipeline>
   |-- <pipeline>.py                # Defines the component logic
   |-- patterns.py                  # Defines matched patterns
   |-- factory.py                   # Declares the pipeline to SpaCy
```

### Style Guide

We use [Black](https://github.com/psf/black) to reformat the code. While other formatter only enforce PEP8 compliance, Black also makes the code uniform. In short :

> Black reformats entire files in place. It is not configurable.

Moreover, the CI/CD pipeline enforces a number of checks on the "quality" of the code. To wit, non black-formatted code will make the test pipeline fail. We use `pre-commit` to keep our codebase clean.

Refer to the [development install tutorial](../home/installation.md) for tips on how to format your files automatically. Most modern editors propose extensions that will format files on save.
