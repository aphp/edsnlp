# Changelog

## v0.3.2

- Major revamp of the normalisation.
  - The `normalizer` pipeline **now adds atomic components** (`lowercase`, `accents`, `quotes`, `pollution` & `endlines`) to the processing pipeline, and compiles the results into a new `Doc._.normalized` extension. The latter is itself a Spacy `Doc` object, wherein tokens are normalised and pollution tokens are removed altogether. Components that match on the `CUSTOM_NORM` attribute process the `normalized` document, and matches are brought back to the original document using a token-wise mapping.
  - Update the `RegexMatcher` to use the `CUSTOM_NORM` attribute
  - Add an `EDSPhraseMatcher`, wrapping Spacy's `PhraseMatcher` to enable matching on `CUSTOM_NORM`.
  - Update the `matcher` and `advanced` pipelines to enable matching on the `CUSTOM_NORM` attribute.
- Add an OMOP connector, to help go back and forth between OMOP-formatted pandas dataframes and Spacy documents.
- Add a `reason` pipeline, that extracts the reason for visit.
- Add an `endlines` pipeline, that classifies newline characters between spaces and actual ends of line.
- Add possibility to annotate within entities for qualifiers (`negation`, `hypothesis`, etc), ie if the cue is within the entity. Disabled by default.

## v0.3.1

- Update `dates` to remove miscellaneous bugs.
- Add `isort` pre-commit hook.
- Improve performance for `negation`, `hypothesis`, `antecedents`, `family` and `rspeech` by using Spacy's `filter_spans` and our `consume_spans` methods.
- Add proposition segmentation to `hypothesis` and `family`, enhancing results.

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
