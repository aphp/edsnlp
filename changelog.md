# Changelog

## v0.4.4

- Add `measures` pipeline
- Cap Jinja2 version to fix mkdocs
- Adding the possibility to add context in the processing module
- Improve the speed of char replacement pipelines (accents and quotes)
- Improve the speed of the regex matcher

## v0.4.3

- Fix regex matching on spans.

## v0.4.2

- Fix issue with `dateparser` library (see scrapinghub/dateparser#1045)
- Fix `attr` issue in the `advanced-regex` pipelin
- Add documentation for `eds.covid`
- Update the demo with an explanation for the regex

## v0.4.1

- Added support to Koalas DataFrames in the `edsnlp.processing` pipe.
- Added `eds.covid` NER pipeline for detecting COVID19 mentions.

## v0.4.0

- Profound re-write of the normalisation :
  - The custom attribute `CUSTOM_NORM` is completely abandoned in favour of a more _spacyfic_ alternative
  - The `normalizer` pipeline modifies the `NORM` attribute in place
  - Other pipelines can modify the `Token._.excluded` custom attribute
- EDS regex and term matchers can ignore excluded tokens during matching, effectively adding a second dimension to normalisation (choice of the attribute and possibility to skip _pollution_ tokens regardless of the attribute)
- Matching can be performed on custom attributes more easily
- Qualifiers are regrouped together within the `edsnlp.qualifiers` submodule, the inheritance from the `GenericMatcher` is dropped.
- `edsnlp.utils.filter.filter_spans` now accepts a `label_to_remove` parameter. If set, only corresponding spans are removed, along with overlapping spans. Primary use-case: removing pseudo cues for qualifiers.
- Generalise the naming convention for extensions, which keep the same name as the pipeline that created them (eg `Span._.negation` for the `eds.negation` pipeline). The previous convention is kept for now, but calling it issues a warning.
- The `dates` pipeline underwent some light formatting to increase robustness and fix a few issues
- A new `consultation_dates` pipeline was added, which looks for dates preceded by expressions specific to consultation dates
- In rule-based processing, the `terms.py` submodule is replaced by `patterns.py` to reflect the possible presence of regular expressions
- Refactoring of the architecture :
  - pipelines are now regrouped by type (`core`, `ner`, `misc`, `qualifiers`)
  - `matchers` submodule contains `RegexMatcher` and `PhraseMatcher` classes, which interact with the normalisation
  - `multiprocessing` submodule contains `spark` and `local` multiprocessing tools
  - `connectors` contains `Brat`, `OMOP` and `LabelTool` connectors
  - `utils` contains various utilities
- Add entry points to make pipeline usable directly, removing the need to import `edsnlp.components`.
- Add a `eds` namespace for components: for instance, `negation` becomes `eds.negation`. Using the former pipeline name still works, but issues a deprecation warning.
- Add 3 score pipelines related to emergency
- Add a helper function to use a spaCy pipeline as a Spark UDF.
- Fix alignment issues in RegexMatcher
- Change the alignment procedure, dropping clumsy `numpy` dependency in favour of `bisect`
- Change the name of `eds.antecedents` to `eds.history`.
  Calling `eds.antecedents` still works, but issues a deprecation warning and support will be removed in a future version.
- Add a `eds.covid` component, that identifies mentions of COVID
- Change the demo, to include NER components

## v0.3.2

- Major revamp of the normalisation.
  - The `normalizer` pipeline **now adds atomic components** (`lowercase`, `accents`, `quotes`, `pollution` & `endlines`) to the processing pipeline, and compiles the results into a new `Doc._.normalized` extension. The latter is itself a spaCy `Doc` object, wherein tokens are normalised and pollution tokens are removed altogether. Components that match on the `CUSTOM_NORM` attribute process the `normalized` document, and matches are brought back to the original document using a token-wise mapping.
  - Update the `RegexMatcher` to use the `CUSTOM_NORM` attribute
  - Add an `EDSPhraseMatcher`, wrapping spaCy's `PhraseMatcher` to enable matching on `CUSTOM_NORM`.
  - Update the `matcher` and `advanced` pipelines to enable matching on the `CUSTOM_NORM` attribute.
- Add an OMOP connector, to help go back and forth between OMOP-formatted pandas dataframes and spaCy documents.
- Add a `reason` pipeline, that extracts the reason for visit.
- Add an `endlines` pipeline, that classifies newline characters between spaces and actual ends of line.
- Add possibility to annotate within entities for qualifiers (`negation`, `hypothesis`, etc), ie if the cue is within the entity. Disabled by default.

## v0.3.1

- Update `dates` to remove miscellaneous bugs.
- Add `isort` pre-commit hook.
- Improve performance for `negation`, `hypothesis`, `antecedents`, `family` and `rspeech` by using spaCy's `filter_spans` and our `consume_spans` methods.
- Add proposition segmentation to `hypothesis` and `family`, enhancing results.

## v0.3.0

- Renamed `generic` to `matcher`. This is a non-breaking change for the average user, adding the pipeline is still :

  <!-- no-check -->

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
