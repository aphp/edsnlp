# Contributing to EDS-NLP

We welcome contributions ! Fork the project and propose a pull request.

At the very least, your changes should :

- Be well-documented ;
- Pass every tests, and preferably implement its own ;
- Follow the style guide.

## Testing your code

We use the Pytest test suite.

The following command will run the test suite. Writing your own tests is encouraged !

```shell script
python -m pytest
```

Should your contribution propose a bug fix, we require the bug be thoroughly tested.

## Style Guide

We use [Black](https://github.com/psf/black) to reformat the code. While other formatter only enforce PEP8 compliance, Black also makes the code uniform. In short :

> Black reformats entire files in place. It is not configurable.

## Architecture of a pipeline

All pipelines should follow the same pattern :

```
edsnlp/pipelines/<pipeline>
   |-- <pipeline>.py                # Defines the component logic
   |-- terms.py                     # Defines matched patterns
   |-- factory.py                   # Declares the pipeline to Spacy
```

Supplementary modules may also be included. To make reproducibility possible, legacy implementations should live in a `leg` sub-module. `<pipeline>` always maps to the latest implementation, and older versions can be retrieved using `.v<version-number>` suffix, eg `<pipeline>.v0`.
