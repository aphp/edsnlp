# Changelog

## Unreleased

### Changed

- Disable `EDSMatcher` preprocessing auto progress tracking by default

## v0.7.4 (2022-12-12)

### Added
- `eds.history` : Add the option to consider only the closest dates in the sentence (dates inside the boundaries and if there is not, it takes the closest date in the entire sentence).
- `eds.negation` : It takes into account following past participates and preceding infinitives.
- `eds.hypothesis`: It takes into account following past participates hypothesis verbs.
- `eds.negation` & `eds.hypothesis` : Introduce new patterns and remove unnecessary patterns.
- `eds.dates` : Add a pattern for preceding relative dates (ex: l'embolie qui est survenue **à 10 jours**).
- Improve patterns in the `eds.pollution` component to account for multiline footers
- Add `QuickExample` object to quickly try a pipeline.
- Add UMLS terminology matcher `eds.umls`
- New `RegexMatcher` method to create spans from groupdicts
- New `eds.dates` option to disable time detection

### Changed

- Improve date detection by removing false positives

### Fixed

- `eds.hypothesis` : Remove too generic patterns.
- `EDSTokenizer` : It now tokenizes `"rechereche d'"` as `["recherche", "d'"]`, instead of `["recherche", "d", "'"]`.
- Fix small typos in the documentation and in the docstring.
- Harmonize processing utils (distributed custom_pipe) to have the same API for Pandas and Pyspark
- Fix BratConnector file loading issues with complex file hierarchies

## v0.7.2 (2022-10-26)

### Added

- Improve the `eds.history` component by taking into account the date extracted from `eds.dates` component.
- New pop up when you click on the copy icon in the termynal widget (docs).
- Add NER `eds.elston-ellis` pipeline to identify Elston Ellis scores
- Add flags=re.MULTILINE to `eds.pollution` and change pattern of footer

### Fixed

- Remove the warning in the ``eds.sections`` when ``eds.normalizer`` is in the pipe.
- Fix filter_spans for strictly nested entities
- Fill eds.remove-lowercase "assign" metadata to run the pipeline during EDSPhraseMatcher preprocessing
- Allow back spaCy components whose name contains a dot (forbidden since spaCy v3.4.2) for backward compatibility.

## v0.7.1 (2022-10-13)

### Added

- Add new patterns (footer, web entities, biology tables, coding sections) to pipeline normalisation (pollution)

### Changed

- Improved TNM detection algorithm
- Account for more modifiers in ADICAP codes detection

### Fixed

- Add nephew, niece and daughter to family qualifier patterns
- EDSTokenizer (`spacy.blank('eds')`) now recognizes non-breaking whitespaces as spaces and does not split float numbers
- `eds.dates` pipeline now allows new lines as space separators in dates

## v0.7.0 (2022-09-06)

### Added

- New nested NER trainable `nested_ner` pipeline component
- Support for nested entities and attributes in BratDataConnector
- Pytorch wrappers and experimental training utils
- Add attribute `section` to entities
- Add new cases for separator pattern when components of the TNM score are separated by a forward slash
- Add NER `eds.adicap` pipeline to identify ADICAP codes
- Add patterns to `pollution` pipeline and simplifies activating or deactivating specific patterns

### Changed
- Simplified the configuration scheme of the `pollution` pipeline
- Update of the `ContextualMatcher` (and all pipelines depending on it), rendering it more flexible to use
- Rename R component of score TNM as "resection_completeness"

### Fixed

- Prevent section titles from capturing surrounding tokens, causing overlaps (#113)
- Enhance existing patterns for section detection and add patterns for previously ignored sections (introduction, evolution, modalites de sortie, vaccination) .
- Fix explain mode, which was always triggered, in `eds.history` factory.
- Fix test in `eds.sections`. Previously, no check was done
- Remove SOFA scores spurious span suffixes

## v0.6.2 (2022-08-02)

### Added

- New `SimstringMatcher` matcher to perform fuzzy term matching, and `algorithm` parameter in terminology components and `eds.matcher` component
- Makefile to install,test the application and see the documentation

### Changed

- Add consultation date pattern "CS", and False Positive patterns for dates (namely phone numbers and pagination).
- Update the pipeline score `eds.TNM`. Now it is possible to return a dictionary where the results are either `str` or `int` values

### Fixed

- Add new patterns to the negation qualifier
- Numpy header issues with binary distributed packages
- Simstring dependency on Windows

## v0.6.1 (2022-07-11)

### Added

- Now possible to provide regex flags when using the RegexMatcher
- New `ContextualMatcher` pipe, aiming at replacing the `AdvancedRegex` pipe.
- New `as_ents` parameter for `eds.dates`, to save detected dates as entities

### Changed

- Faster `eds.sentences` pipeline component with Cython
- Bump version of Pydantic in `requirements.txt` to 1.8.2 to handle an incompatibility with the ContextualMatcher
- Optimise space requirements by using `.csv.gz` compression for verbs

### Fixed

- `eds.sentences` behaviour with dot-delimited dates (eg `02.07.2022`, which counted as three sentences)

## v0.6.0 (2022-06-17)

### Added

- Complete revamp of the measurements detection pipeline, with better parsing and more exhaustive matching
- Add new functionality to the method `Span._.date.to_datetime()` to return a result infered from context for those cases with missing information.
- Force a batch size of 2000 when distributing a pipeline with Spark
- New patterns to pipeline `eds.dates` to identify cases where only the month is mentioned
- New `eds.terminology` component for generic terminology matching, using the `kb_id_` attribute to store fine-grained entity label
- New `eds.cim10` terminology matching pipeline
- New `eds.drugs` terminology pipeline that maps brand names and active ingredients to a unique [ATC](https://en.wikipedia.org/wiki/Anatomical_Therapeutic_Chemical_Classification_System) code

## v0.5.3 (2022-05-04)

### Added

- Support for strings in the example utility
- [TNM](https://en.wikipedia.org/wiki/TNM_staging_system) detection and normalisation with the `eds.TNM` pipeline
- Support for arbitrary callback for Pandas multiprocessing, with the `callback` argument

## v0.5.2 (2022-04-29)

### Added

- Support for chained attributes in the `processing` pipelines
- Colour utility with the category20 colour palette

### Fixed

- Correct a REGEX on the date detector (both `nov` and `nov.` are now detected, as all other months)

## v0.5.1 (2022-04-11)

### Fixed

- Updated Numpy requirements to be compatible with the `EDSPhraseMatcher`

## v0.5.0 (2022-04-08)

### Added

- New `eds` language to better fit French clinical documents and improve speed
- Testing for markdown codeblocks to make sure the documentation is actually executable

### Changed

- Complete revamp of the date detection pipeline, with better parsing and more exhaustive matching
- Reimplementation of the EDSPhraseMatcher in Cython, leading to a x15 speed increase

## v0.4.4

- Add `measures` pipeline
- Cap Jinja2 version to fix mkdocs
- Adding the possibility to add context in the processing module
- Improve the speed of char replacement pipelines (accents and quotes)
- Improve the speed of the regex matcher

## v0.4.3

- Fix regex matching on spans.
- Add fast_parse in date pipeline.
- Add relative_date information parsing

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
