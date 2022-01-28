# Contributing to EDS-NLP

We welcome contributions ! There are many ways to help. For example, you can:

1. Develop a new pipeline ! Fork the project and propose a new functionality through a merge request ;
2. Help us track issues through bug reports ;
3. Suggest and help prioritise new functionalities, without necessarily taking part in active development ;
4. Help us make the library as straightforward as possible, by simply asking questions on whatever does not seem clear to you.

We use a dedicated [collaboration board](https://gitlab.eds.aphp.fr/datasciencetools/edsnlp/-/boards/364) to track questions, suggestions, etc. Issues filed on the board can be one of three main types:

| Type       | Comment                                 |
| ---------- | --------------------------------------- |
| Bug report | Report a bug !                          |
| Bug report | Suggest a new feature                   |
| Bug report | General-purpose question on the library |

## Proposing a merge request

At the very least, your changes should :

- Be well-documented ;
- Pass every tests, and preferably implement its own ;
- Follow the style guide.

### Testing your code

We use the Pytest test suite.

The following command will run the test suite. Writing your own tests is encouraged !

```shell script
python -m pytest
```

Should your contribution propose a bug fix, we require the bug be thoroughly tested.

### Architecture of a pipeline

All pipelines should follow the same pattern :

```
edsnlp/pipelines/<pipeline>
   |-- <pipeline>.py                # Defines the component logic
   |-- terms.py                     # Defines matched patterns
   |-- factory.py                   # Declares the pipeline to Spacy
```

Supplementary modules may also be included. To make reproducibility possible, legacy implementations should live in a `leg` sub-module. `<pipeline>` always maps to the latest implementation, and older versions can be retrieved using `.v<version-number>` suffix, eg `<pipeline>.v0`.

### Style Guide

We use [Black](https://github.com/psf/black) to reformat the code. While other formatter only enforce PEP8 compliance, Black also makes the code uniform. In short :

> Black reformats entire files in place. It is not configurable.

Moreover, the CI/CD pipeline (see [`.gitlab-ci.yml`](https://gitlab.eds.aphp.fr/datasciencetools/edsnlp/-/blob/master/.gitlab-ci.yml)) enforces a number of checks on the "quality" of the code. To wit, non black-formatted code will make the test pipeline fail.

Refer to the [development install tutorial](../getting-started/installation.md) for tips on how to format your files automatically. Most modern editors propose extensions that will format files on save.
