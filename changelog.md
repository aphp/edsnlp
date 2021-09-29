# Changelog

## v0.3.0

- Renamed `generic` to `matcher`. This is a non-breaking change for the average user, adding the pipeline is still :
  ```python
  nlp.add_pipe("matcher", config=dict(terms=dict(maladie="maladie")))
  ```
- Removed `quickumls` pipeline. It was untested, unmaintained. Will be added back in a future release.
- Add `score` pipeline, and `charlson`.
- Add `advanced-regex` pipeline
- Corrected bugs in the `negation` pipeline

## v0.2.0

- Add `negation` pipeline
- Add `family` pipeline
- Add `hypothesis` pipeline
- Add `antecedents` pipeline
- Add `rspeech` pipeline
- Refactor the library :
  - Remove the `rules` folder
  - Add a `pipelines` folder, containing one subdirectory per component
  - Every component subdirectory contains a module defining the component, and a module defining a factory, plus any other utilities (eg `terms.py`)

## v0.1.0

First working version. Available pipelines :

- `section`
- `sentences`
- `normalization`
- `pollution`
