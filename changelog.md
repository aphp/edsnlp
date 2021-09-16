# Changelog

## v0.1.0

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
