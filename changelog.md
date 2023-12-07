# Changelog

## Unreleased

### Fixed

- Microgram scale is now correctly 1/1000, instead of 1/10mg before

## v0.10.0

### Added

- New add unified `edsnlp.data` api (json, brat, spark, pandas) and LazyCollection object
  to efficiently read / write data from / to different formats & sources.
- New unified processing API to select the execution execution backends via `data.set_processing(...)`
- The training scripts can now use data from multiple concatenated adapters
- Support quantized transformers (compatible with multiprocessing as well !)

### Changed

- `edsnlp.pipelines` has been renamed to `edsnlp.pipes`, but the old name is still available for backward compatibility
- Pipes (in `edsnlp/pipes`) are now lazily loaded, which should improve the loading time of the library.
- `to_disk` methods can now return a config to override the initial config of the pipeline (e.g., to load a transformer directly from the path storing its fine-tuned weights)
- The `eds.tokenizer` tokenizer has been added to entry points, making it accessible from the outside
- Deprecate old connectors (e.g. BratDataConnector) in favor of the new `edsnlp.data` API
- Deprecate old `pipe` wrapper in favor of the new processing API

### Fixed

- Support for pydantic v2
- Support for python 3.11 (not ci-tested yet)

## v0.10.0beta1

Large refacto of EDS-NLP to allow training models and performing inference using PyTorch
as the deep-learning backend. Rather than a mere wrapper of Pytorch using spaCy, this is
a new framework to build hybrid multi-task models.

To achieve this, instead of patching spaCy's pipeline, a new pipeline was implemented in
a similar fashion to aphp/edspdf#12. The new pipeline tries to preserve the existing API,
especially for non-machine learning uses such as rule-based components. This means that
users can continue to use the library in the same way as before, while also having the option to train models using PyTorch. We still
use spaCy data structures such as Doc and Span to represent the texts and their annotations.

Otherwise, changes should be transparent for users that still want to use spacy pipelines
with `nlp = spacy.blank('eds')`. To benefit from the new features, users should use
`nlp = edsnlp.blank('eds')` instead.

### Added

- New pipeline system available via `edsnlp.blank('eds')` (instead of `spacy.blank('eds')`)
- Use the confit package to instantiate components
- Training script with Pytorch only (`tests/training/`) and tutorial
- New trainable embeddings: `eds.transformer`, `eds.text_cnn`, `eds.span_pooler`
  embedding contextualizer pipes
- Re-implemented the trainable NER component and trainable Span qualifier with the new
  system under `eds.ner_crf` and `eds.span_classifier`
- New efficient implementation for eds.transformer (to be used in place of
  spacy-transformer)

### Changed

- Pipe registering: `Language.factory` -> `edsnlp.registry.factory.register` via confit
- Lazy loading components from their entry point (had to patch spacy.Language.__init__)
  to avoid having to wrap every import torch statement for pure rule-based use cases.
  Hence, torch is not a required dependency

## v0.9.2

### Changed

- Fix matchers to skip pipes with assigned extensions that are not required by the matcher during the initialization

## v0.9.1

### Changed

- Improve negation patterns
- Abstent disorders now set the negation to True when matched as `ABSENT`
- Default qualifier is now `None` instead of `False` (empty string)

### Fixed

- `span_getter` is not incompatible with on_ents_only anymore
- `ContextualMatcher` now supports empty matches (e.g. lookahead/lookbehind) in `assign` patterns

## v0.9.0

### Added

- New `to_duration` method to convert an absolute date into a date relative to the note_datetime (or None)

### Changes

- Input and output of components are now specified by `span_getter` and `span_setter` arguments.
- :boom: Score / disorders / behaviors entities now have a fixed label (passed as an argument), instead of being dynamically set from the component name. The following scores may have a different name than the current one in your pipelines:
  * `eds.emergency.gemsa` → `emergency_gemsa`
  * `eds.emergency.ccmu` → `emergency_ccmu`
  * `eds.emergency.priority` → `emergency_priority`
  * `eds.charlson` → `charlson`
  * `eds.elston_ellis` → `elston_ellis`
  * `eds.SOFA` → `sofa`
  * `eds.adicap` → `adicap`
  * `eds.measuremets` → `size`, `weight`, ... instead of `eds.size`, `eds.weight`, ...
- `eds.dates` now separate dates from durations. Each entity has its own label:
  * `spans["dates"]` → entities labelled as `date` with a `span._.date` parsed object
  * `spans["durations"]` → entities labelled as `duration` with a `span._.duration` parsed object
- the "relative" / "absolute" / "duration" mode of the time entity is now stored in
  the `mode` attribute of the `span._.date/duration`
- the "from" / "until" period bound, if any, is now stored in the `span._.date.bound` attribute
- `to_datetime` now only return absolute dates, converts relative dates into absolute if `doc._.note_datetime` is given, and None otherwise

### Fixed
- `export_to_brat` issue with spans of entities on multiple lines.

## v0.8.1 (2023-05-31)

Fix release to allow installation from source

## v0.8.0 (2023-05-24)

### Added

- New trainable component for multi-label, multi-class span qualification (any attribute/extension)
- Add range measurements (like `la tumeur fait entre 1 et 2 cm`) to `eds.measurements` matcher
- Add `eds.CKD` component
- Add `eds.COPD` component
- Add `eds.alcohol` component
- Add `eds.cerebrovascular_accident` component
- Add `eds.congestive_heart_failure` component
- Add `eds.connective_tissue_disease` component
- Add `eds.dementia` component
- Add `eds.diabetes` component
- Add `eds.hemiplegia` component
- Add `eds.leukemia` component
- Add `eds.liver_disease` component
- Add `eds.lymphoma` component
- Add `eds.myocardial_infarction` component
- Add `eds.peptic_ulcer_disease` component
- Add `eds.peripheral_vascular_disease` component
- Add `eds.solid_tumor` component
- Add `eds.tobacco` component
- Add `eds.spaces` (or `eds.normalizer` with `spaces=True`) to detect space tokens, and add `ignore_space_tokens` to `EDSPhraseMatcher` and `SimstringMatcher` to skip them
- Add `ignore_space_tokens` option in most components
- `eds.tables`: new pipeline to identify formatted tables
- New `merge_mode` parameter in `eds.measurements` to normalize existing entities or detect
  measures only inside existing entities
- Tokenization exceptions (`Mr.`, `Dr.`, `Mrs.`) and non end-of-sentence periods are now tokenized with the next letter in the `eds` tokenizer

### Changed

- Disable `EDSMatcher` preprocessing auto progress tracking by default
- Moved dependencies to a single pyproject.toml: support for `pip install -e '.[dev,docs,setup]'`
- ADICAP matcher now allow dot separators (e.g. `B.H.HP.A7A0`)

### Fixed

- Abbreviation and number tokenization issues in the `eds` tokenizer
- `eds.adicap` : reparsed the dictionnary used to decode the ADICAP codes (some of them were wrongly decoded)
- Fix build for python 3.9 on Mac M1/M2 machines.

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

  ```{ .python .no-check }
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
