# Changelog

# Unreleased

### Added

- `edsnlp.data.read_parquet` now accept a `work_unit="fragment"` option to split tasks between workers by parquet fragment instead of row. When this is enabled, workers do not read every fragment while skipping 1 in n rows, but read all rows of 1/n fragments, which should be faster.
- Accept no validation data in `edsnlp.train` script
- Log the training config at the beginning of the trainings
- Support a specific model output dir path for trainings (`output_model_dir`), and whether to save the model or not (`save_model`)
- Specify whether to log the validation results or not (`logger=False`)
- Added support for the CoNLL format with `edsnlp.data.read_conll` and with a specific `eds.conll_dict2doc` converter
- Added a Trainable Biaffine Dependency Parser (`eds.biaffine_dep_parser`) component and metrics
- New `eds.extractive_qa` component to perform extractive question answering using questions as prompts to tag entities instead of a list of predefined labels as in `eds.ner_crf`.

### Fixed

- Fix `join_thread` missing attribute in `SimpleQueue` when cleaning a multiprocessing executor
- Support huggingface transformers that do not set `cls_token_id` and `sep_token_id` (we now also look for these tokens in the `special_tokens_map` and `vocab` mappings)
- Fix changing scorers dict size issue when evaluating during training
- Seed random states (instead of using `random.RandomState()`) when shuffling in data readers : this is important for
  1. reproducibility
  2. in multiprocessing mode, ensure that the same data is shuffled in the same way in all workers
- Bubble BaseComponent instantiation errors correctly
- Improved support for multi-gpu gradient accumulation (only sync the gradients at the end of the accumulation), now controled by the optiona `sub_batch_size` argument of `TrainingData`.
- Support again edsnlp without pytorch installed
- We now test that edsnlp works without pytorch installed

## v0.14.0 (2024-11-14)

### Added

- Support for setuptools based projects in `edsnlp.package` command
- Pipelines can now be instantiated directly from a config file (instead of having to cast a dict containing their arguments) by putting the @core = "pipeline" or "load" field in the pipeline section)
- `edsnlp.load` now correctly takes disable, enable and exclude parameters into account
- Pipeline now has a basic repr showing is base langage (mostly useful to know its tokenizer) and its pipes
- New `python -m edsnlp.evaluate` script to evaluate a model on a dataset
- Sentence detection can now be configured to change the minimum number of newlines to consider a newline-triggered sentence, and disable capitalization checking.
- New `eds.split` pipe to split a document into multiple documents based on a splitting pattern (useful for training)
- Allow `converter` argument of `edsnlp.data.read/from_...` to be a list of converters instead of a single converter
- New revamped and documented `edsnlp.train` script and API
- Support YAML config files (supported only CFG/INI files before)
- Most of EDS-NLP functions are now clickable in the documentation
- ScheduledOptimizer now accepts schedules directly in place of parameters, and easy parameter selection:

    ```
    ScheduledOptimizer(
        optim="adamw",
        module=nlp,
        total_steps=2000,
        groups={
            "^transformer": {
                # lr will go from 0 to 5e-5 then to 0 for params matching "transformer"
                "lr": {"@schedules": "linear", "warmup_rate": 0.1, "start_value": 0 "max_value": 5e-5,},
            },
            "": {
                # lr will go from 3e-4 during 200 steps then to 0 for other params
                "lr": {"@schedules": "linear", "warmup_rate": 0.1, "start_value": 3e-4 "max_value": 3e-4,},
            },
        },
    )
    ```

### Changed

- `eds.span_context_getter`'s parameter `context_sents` is no longer optional and must be explicitly set to 0 to disable sentence context
- In multi-GPU setups, streams that contain torch components are now stripped of their parameter tensors when sent to CPU Workers since these workers only perform preprocessing and postprocessing and should therefore not need the model parameters.
- The `batch_size` argument of `Pipeline` is deprecated and is not used anymore. Use the `batch_size` argument of `stream.map_pipeline` instead.

### Fixed

- Sort files before iterating over a standoff or json folder to ensure reproducibility
- Sentence detection now correctly match capitalized letters + apostrophe
- We now ensure that the workers pool is properly closed whatever happens (exception, garbage collection, data ending) in the `multiprocessing` backend. This prevents some executions from hanging indefinitely at the end of the processing.
- Propagate torch sharing strategy to other workers in the `multiprocessing` backend. This is useful when the system is running out of file descriptors and `ulimit -n` is not an option. Torch sharing strategy can also be set via an environment variable `TORCH_SHARING_STRATEGY` (default is `file_descriptor`, [consider using `file_system` if you encounter issues](https://pytorch.org/docs/stable/multiprocessing.html#file-system-file-system)).

### Data API changes

- `LazyCollection` objects are now called `Stream` objects
- By default, `multiprocessing` backend now preserves the order of the input data. To disable this and improve performance, use `deterministic=False` in the `set_processing` method
- :rocket: Parallelized GPU inference throughput improvements !

    - For simple {pre-process → model → post-process} pipelines, GPU inference can be up to 30% faster in non-deterministic mode (results can be out of order) and up to 20% faster in deterministic mode (results are in order)
    - For multitask pipelines, GPU inference can be up to twice as fast (measured in a two-tasks BERT+NER+Qualif pipeline on T4 and A100 GPUs)

- The `.map_batches`, `.map_pipeline` and `.map_gpu` methods now support a specific `batch_size` and batching function, instead of having a single batch size for all pipes
- Readers now have a `loop` parameter to cycle over the data indefinitely (useful for training)
- Readers now have a `shuffle` parameter to shuffle the data before iterating over it
- In `multiprocessing` mode, file based readers now read the data in the workers (was an option before)
- We now support two new special batch sizes

    - "fragment" in the case of parquet datasets: rows of a full parquet file fragment per batch
    - "dataset" which is mostly useful during training, for instance to shuffle the dataset at each epoch.
  These are also compatible in batched writer such as parquet, where each input fragment can be processed and mapped to a single matching output fragment.

- :boom: Breaking change: a `map` function returning a list or a generator won't be automatically flattened anymore. Use `flatten()` to flatten the output if needed. This shouldn't change the behavior for most users since most writers (to_pandas, to_polars, to_parquet, ...) still flatten the output
- :boom: Breaking change: the `chunk_size` and `sort_chunks` are now deprecated : to sort data before applying a transformation, use `.map_batches(custom_sort_fn, batch_size=...)`

### Training API changes

- We now provide a training script `python -m edsnlp.train --config config.cfg` that should fit many use cases. Check out the docs !
- In particular, we do not require pytorch's Dataloader for training and can rely solely on EDS-NLP stream/data API, which is better suited for large streamable datasets and dynamic preprocessing (ie different result each time we apply a noised preprocessing op on a sample).
- Each trainable component can now provide a `stats` field in its `preprocess` output to log info about the sample (number of words, tokens, spans, ...):

    - these stats are both used for batching (e.g., make batches of no more than "25000 tokens")
    - for logging
    - for computing correct loss means when accumulating gradients over multiple mini-mini-batches
    - for computing correct loss means in multi-GPU setups, since these stats are synchronized and accumulated across GPUs

- Support multi GPU training via hugginface `accelerate` and EDS-NLP `Stream` API consideration of env['WOLRD_SIZE'] and env['LOCAL_RANK'] environment variables

## v0.13.1

### Added

- `eds.tables` accepts a minimum_table_size (default 2) argument to reduce pollution
- `RuleBasedQualifier` now expose a `process` method that only returns qualified entities and token without actually tagging them, deferring this task to the `__call__` method.
- Added new patterns for metastasis detection. Developed on CT-Scan reports.
- Added citation of articles

### Changed

- Renamed `edsnlp.scorers` to `edsnlp.metrics` and removed the `_scorer` suffix from their
  registry name (e.g, `@scorers = ner_overlap_scorer` → `@metrics = ner_overlap`)
- Rename `eds.measurements` to `eds.quantities`
- scikit-learn (used in `eds.endlines`) is no longer installed by default when installing `edsnlp[ml]`

### Fixed

- Disorder and Behavior pipes don't use a "PRESENT" or "ABSENT" `status` anymore. Instead, `status=None` by default,
  and `ent._.negation` is set to True instead of setting `status` to "ABSENT". To this end, the *tobacco* and *alcohol*
  now use the `NegationQualifier` internally.
- Numbers are now only detected without trying to remove the pollution in between digits, ie `55 @ 77777` could be detected as a full number before, but not anymore.
- Resolve encoding-related data reading issues by forcing utf-8

## v0.13.0

### Added

- `data.set_processing(...)` now expose an `autocast` parameter to disable or tweak the automatic casting of the tensor
  during the processing. Autocasting should result in a slight speedup, but may lead to numerical instability.
- Use `torch.inference_mode` to disable view tracking and version counter bumps during inference.
- Added a new NER pipeline for suicide attempt detection
- Added date cues (regular expression matches that contributed to a date being detected) under the extension `ent._.date_cues`
- Added tables processing in eds.measurement
- Added 'all' as possible input in eds.measurement measurements config
- Added new units in eds.measurement

### Changed

- Default to mixed precision inference

### Fixed

- `edsnlp.load("your/huggingface-model", install_dependencies=True)` now correctly resolves the python pip
  (especially on Colab) to auto-install the model dependencies
- We now better handle empty documents in the `eds.transformer`, `eds.text_cnn` and `eds.ner_crf` components
- Support mixed precision in `eds.text_cnn` and `eds.ner_crf` components
- Support pre-quantization (<4.30) transformers versions
- Verify that all batches are non empty
- Fix `span_context_getter` for `context_words` = 0, `context_sents` > 2 and support assymetric contexts
- Don't split sentences on rare unicode symbols
- Better detect abbreviations, like `E.coli`, now split as [`E.`, `coli`] and not [`E`, `.`, `coli`]

## v0.12.3

### Changed

Packages:

- Pip-installable models are now built with `hatch` instead of poetry, which allows us to expose `artifacts` (weights)
  at the root of the sdist package (uploadable to HF) and move them inside the package upon installation to avoid conflicts.
- Dependencies are no longer inferred with dill-magic (this didn't work well before anyway)
- Option to perform substitutions in the model's README.md file (e.g., for the model's name, metrics, ...)
- Huggingface models are now installed with pip *editable* installations, which is faster since it doesn't copy around the weights

## v0.12.1

### Added

- Added binary distribution for linux aarch64 (Streamlit's environment)
- Added new separator option in eds.table and new input check

### Fixed

- Make catalogue & entrypoints compatible with py37-py312
- Check that a data has a doc before trying to use the document's `note_datetime`

## v0.12.0

### Added

- The `eds.transformer` component now accepts `prompts` (passed to its `preprocess` method, see breaking change below) to add before each window of text to embed.
- `LazyCollection.map` / `map_batches` now support generator functions as arguments.
- Window stride can now be disabled (i.e., stride = window) during training in the `eds.transformer` component by `training_stride = False`
- Added a new `eds.ner_overlap_scorer` to evaluate matches between two lists of entities, counting true when the dice overlap is above a given threshold
- `edsnlp.load` now accepts EDS-NLP models from the huggingface hub 🤗 !
- New `python -m edsnlp.package` command to package a model for the huggingface hub or pypi-like registries
- Improve table detection in `eds.tables` and support new options in `table._.to_pd_table(...)`:
  - `header=True` to use first row as header
  - `index=True` to use first column as index
  - `as_spans=True` to fill cells as document spans instead of strings

### Changed

- :boom: Major breaking change in trainable components, moving towards a more "task-centric" design:
  - the `eds.transformer` component is no longer responsible for deciding which spans of text ("contexts") should be embedded. These contexts are now passed via the `preprocess` method, which now accepts more arguments than just the docs to process.
  - similarly the `eds.span_pooler` is now longer responsible for deciding which spans to pool, and instead pools all spans passed to it in the `preprocess` method.

  Consequently, the `eds.transformer` and `eds.span_pooler` no longer accept their `span_getter` argument, and the `eds.ner_crf`, `eds.span_classifier`, `eds.span_linker` and `eds.span_qualifier` components now accept a `context_getter` argument instead, as well as a `span_getter` argument for the latter two. This refactoring can be summarized as follows:

    ```diff
    - eds.transformer.span_getter
    + eds.ner_crf.context_getter
    + eds.span_classifier.context_getter
    + eds.span_linker.context_getter

    - eds.span_pooler.span_getter
    + eds.span_qualifier.span_getter
    + eds.span_linker.span_getter
    ```

    and as an example for the `eds.span_linker` component:

    ```diff
    nlp.add_pipe(
        eds.span_linker(
            metric="cosine",
            probability_mode="sigmoid",
    +       span_getter="ents",
    +       # context_getter="ents",  -> by default, same as span_getter
            embedding=eds.span_pooler(
                hidden_size=128,
    -           span_getter="ents",
                embedding=eds.transformer(
    -               span_getter="ents",
                    model="prajjwal1/bert-tiny",
                    window=128,
                    stride=96,
                ),
            ),
        ),
        name="linker",
    )
    ```
- Trainable embedding components now all use `foldedtensor` to return embeddings, instead of returning a tensor of floats and a mask tensor.
- :boom: TorchComponent `__call__` no longer applies the end to end method, and instead calls the `forward` method directly, like all torch modules.
- The trainable `eds.span_qualifier` component has been renamed to `eds.span_classifier` to reflect its general purpose (it doesn't only predict qualifiers, but any attribute of a span using its context or not).
- `omop` converter now takes the `note_datetime` field into account by default when building a document
- `span._.date.to_datetime()` and `span._.date.to_duration()` now automatically take the `note_datetime` into account
- `nlp.vocab` is no longer serialized when saving a model, as it may contain sensitive information and can be recomputed during inference anyway

### Fixed

- `edsnlp.data.read_json` now correctly read the files from the directory passed as an argument, and not from the parent directory.
- Overwrite spacy's Doc, Span and Token pickling utils to allow recursively storing Doc, Span and Token objects in the extension values (in particular, span._.date.doc)
- Removed pendulum dependency, solving various pickling, multiprocessing and missing attributes errors

## v0.11.2

### Fixed
- Fix `edsnlp.utils.file_system.normalize_fs_path` file system detection not working correctly
- Improved performance of `edsnlp.data` methods over a filesystem (`fs` parameter)

## v0.11.1 (2024-04-02)

### Added

- Automatic estimation of cpu count when using multiprocessing
- `optim.initialize()` method to create optim state before the first backward pass

### Changed

- `nlp.post_init` will not tee lazy collections anymore (use `edsnlp.utils.collections.multi_tee` yourself if needed)

### Fixed

- Corrected inconsistencies in `eds.span_linker`

## v0.11.0 (2024-03-29)

### Added

- Support for a `filesystem` parameter in every `edsnlp.data.read_*` and `edsnlp.data.write_*` functions
- Pipes of a pipeline are now easily accessible with `nlp.pipes.xxx` instead of `nlp.get_pipe("xxx")`
- Support builtin Span attributes in converters `span_attributes` parameter, e.g.
  ```python
  import edsnlp

  nlp = ...
  nlp.add_pipe("eds.sentences")

  data = edsnlp.data.from_xxx(...)
  data = data.map_pipeline(nlp)
  data.to_pandas(converters={"ents": {"span_attributes": ["sent.text", "start", "end"]}})
  ```
- Support assigning Brat AnnotatorNotes as span attributes: `edsnlp.data.read_standoff(...,  notes_as_span_attribute="cui")`
- Support for mapping full batches in `edsnlp.processing` pipelines with `map_batches` lazy collection method:
  ```python
  import edsnlp

  data = edsnlp.data.from_xxx(...)
  data = data.map_batches(lambda batch: do_something(batch))
  data.to_pandas()
  ```
- New `data.map_gpu` method to map a deep learning operation on some data and take advantage of edsnlp multi-gpu inference capabilities
- Added average precision computation in edsnlp span_classification scorer
- You can now add pipes to your pipeline by instantiating them directly, which comes with many advantages, such as auto-completion, introspection and type checking !

  ```python
  import edsnlp, edsnlp.pipes as eds

  nlp = edsnlp.blank("eds")
  nlp.add_pipe(eds.sentences())
  # instead of nlp.add_pipe("eds.sentences")
  ```

  *The previous way of adding pipes is still supported.*
- New `eds.span_linker` deep-learning component to match entities with their concepts in a knowledge base, in synonym-similarity or concept-similarity mode.

### Changed

- `nlp.preprocess_many` now uses lazy collections to enable parallel processing
- :warning: Breaking change. Improved and simplified `eds.span_qualifier`: we didn't support combination groups before, so this feature was scrapped for now. We now also support splitting values of a single qualifier between different span labels.
- Optimized edsnlp.data batching, especially for large batch sizes (removed a quadratic loop)
- :warning: Breaking change. By default, the name of components added to a pipeline is now the default name defined in their class `__init__` signature. For most components of EDS-NLP, this will change the name from "eds.xxx" to "xxx".

### Fixed

- Flatten list outputs (such as "ents" converter) when iterating: `nlp.map(data).to_iterable("ents")` is now a list of entities, and not a list of lists of entities
- Allow span pooler to choose between multiple base embedding spans (as likely produced by `eds.transformer`) by sorting them by Dice overlap score.
- EDS-NLP does not raise an error anymore when saving a model to an already existing, but empty directory

## v0.10.7 (2024-03-12)

### Added

- Support empty writer converter by default in `edsnlp.data` readers / writers (do not convert by default)
- Add support for polars data import / export
- Allow kwargs in `eds.transformer` to pass to the transformer model

### Changed

- Saving pipelines now longer saves the `disabled` status of the pipes (i.e., all pipes are considered "enabled" when saved). This feature was not used and causing issues when saving a model wrapped in a `nlp.select_pipes` context.

### Fixed

- Allow missing `meta.json`, `tokenizer` and `vocab` paths when loading saved models
- Save torch buffers when dumping machine learning models to disk (previous versions only saved the model parameters)
- Fix automatic `batch_size` estimation in `eds.transformer` when `max_tokens_per_device` is set to `auto` and multiple GPUs are used
- Fix JSONL file parsing

## v0.10.6 (2024-02-24)

### Added

- Added `batch_by`, `split_into_batches_after`, `sort_chunks`, `chunk_size`, `disable_implicit_parallelism` parameters to processing (`simple` and `multiprocessing`) backends to improve performance
  and memory usage. Sorting chunks can improve yield up to **twice the speed** in some cases.
- The deep learning cache mechanism now supports multitask models with weight sharing in multiprocessing mode.
- Added `max_tokens_per_device="auto"` parameter to `eds.transformer` to estimate memory usage and automatically split the input into chunks that fit into the GPU.

### Changed

- Improved speed and memory usage of the `eds.text_cnn` pipe by running the CNN on a non-padded version of its input: expect a speedup up to 1.3x in real-world use cases.
- Deprecate the converters' (especially for BRAT/Standoff data) `bool_attributes`
  parameter in favor of general `default_attributes`. This new mapping describes how to
  set attributes on spans for which no attribute value was found in the input format.
  This is especially useful for negation, or frequent attributes values (e.g. "negated"
  is often False, "temporal" is often "present"), that annotators may not want to
  annotate every time.
- Default `eds.ner_crf` window is now set to 40 and stride set to 20, as it doesn't
  affect throughput (compared to before, window set to 20) and improves accuracy.
- New default `overlap_policy='merge'` option and parameter renaming in
  `eds.span_context_getter` (which replaces `eds.span_sentence_getter`)

### Fixed

- Improved error handling in `multiprocessing` backend (e.g., no more deadlock)
- Various improvements to the data processing related documentation pages
- Begin of sentence / end of sentence transitions of the `eds.ner_crf` component are now
  disabled when windows are used (e.g., neither `window=1` equivalent to softmax and
  `window=0`equivalent to default full sequence Viterbi decoding)
- `eds` tokenizer nows inherits from `spacy.Tokenizer` to avoid typing errors
- Only match 'ne' negation pattern when not part of another word to avoid false positives cases like `u[ne] cure de 10 jours`
- Disabled pipes are now correctly ignored in the `Pipeline.preprocess` method
- Add "eventuel*" patterns to `eds.hyphothesis`

## v0.10.5 (2024-01-29)

### Fixed

- Allow non-url paths when parquet filesystem is given

## v0.10.4 (2024-01-19)

### Changed

- Assigning `doc._.note_datetime` will now automatically cast the value to a `pendulum.DateTime` object

### Added

- Support loading model from package name (e.g., `edsnlp.load("eds_pseudo_aphp")`)
- Support filesystem parameter in `edsnlp.data.read_parquet` and `edsnlp.data.write_parquet`

### Fixed

- Support doc -> list converters with parquet files writer
- Fixed some OOM errors when writing many outputs to parquet files
- Both edsnlp & spacy factories are now listed when a factory lookup fails
- Fixed some GPU OOM errors with the `eds.transformer` pipe when processing really long documents

## v0.10.3 (2024-01-11)

### Added

- By default, `edsnlp.data.write_json` will infer if the data should be written as a single JSONL
  file or as a directory of JSON files, based on the `path` argument being a file or not.

### Fixed

- Measurements now correctly match "0.X", "0.XX", ... numbers
- Typo in "celsius" measurement unit
- Spaces and digits are now supported in BRAT entity labels
- Fixed missing 'permet pas + verb' false positive negation patterns

## v0.10.2 (2023-12-20)

### Changed

- `eds.span_qualifier` qualifiers argument now automatically adds the underscore prefix if not present

### Fixed

- Fix imports of components declared in `spacy_factories` entry points
- Support `pendulum` v3
- `AsList` errors are now correctly reported
- `eds.span_qualifier` saved configuration during `to_disk` is now longer null

## v0.10.1 (2023-12-15)

### Changed

- Small regex matching performance improvement, up to 1.25x faster (e.g. `eds.measurements`)

### Fixed

- Microgram scale is now correctly 1/1000g and inverse meter now 1/100 inverse cm.
- We now isolate some of edsnlp components (trainable pipes that require ml dependencies)
  in a new `edsnlp_factories` entry points to prevent spacy from auto-importing them.
- TNM scores followed by a space are now correctly detected
- Removed various short TNM false positives (e.g., "PT" or "a T") and false negatives
- The Span value extension is not more forcibly overwritten, and user assigned values are returned by `Span._.value` in priority, before the aggregated `span._.get(span.label_)` getter result (#220)
- Enable mmap during multiprocessing model transfers
- `RegexMatcher` now supports all alignment modes (`strict`, `expand`, `contract`) and better handles partial doc matching (#201).
- `on_ent_only=False/True` is now supported again in qualifier pipes (e.g., "eds.negation", "eds.hypothesis", ...)

## v0.10.0 (2023-12-04)

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

## v0.10.0beta1 (2023-12-04)

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

## v0.9.2 (2023-12-04)

### Changed

- Fix matchers to skip pipes with assigned extensions that are not required by the matcher during the initialization

## v0.9.1 (2023-09-22)

### Changed

- Improve negation patterns
- Abstent disorders now set the negation to True when matched as `ABSENT`
- Default qualifier is now `None` instead of `False` (empty string)

### Fixed

- `span_getter` is not incompatible with on_ents_only anymore
- `ContextualMatcher` now supports empty matches (e.g. lookahead/lookbehind) in `assign` patterns

## v0.9.0 (2023-09-15)

### Added

- New `to_duration` method to convert an absolute date into a date relative to the note_datetime (or None)

### Changes

- Input and output of components are now specified by `span_getter` and `span_setter` arguments.
- :boom: Score / disorders / behaviors entities now have a fixed label (passed as an argument), instead of being dynamically set from the component name. The following scores may have a different name
  than the current one in your pipelines:
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

## v0.5.2 (2022-05-04)

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

## v0.4.4 (2022-03-31)

- Add `measures` pipeline
- Cap Jinja2 version to fix mkdocs
- Adding the possibility to add context in the processing module
- Improve the speed of char replacement pipelines (accents and quotes)
- Improve the speed of the regex matcher

## v0.4.3 (2022-03-18)

- Fix regex matching on spans.
- Add fast_parse in date pipeline.
- Add relative_date information parsing

## v0.4.2 (2022-03-16)

- Fix issue with `dateparser` library (see scrapinghub/dateparser#1045)
- Fix `attr` issue in the `advanced-regex` pipelin
- Add documentation for `eds.covid`
- Update the demo with an explanation for the regex

## v0.4.1 (2022-03-14)

- Added support to Koalas DataFrames in the `edsnlp.processing` pipe.
- Added `eds.covid` NER pipeline for detecting COVID19 mentions.

## v0.4.0 (2022-02-22)

- Profound re-write of the normalisation :
    - The custom attribute `CUSTOM_NORM` is completely abandoned in favour of a more _spacyfic_ alternative
    - The `normalizer` pipeline modifies the `NORM` attribute in place
    - Other pipelines can modify the `Token._.excluded` custom attribute
- EDS regex and term matchers can ignore excluded tokens during matching, effectively adding a second dimension to normalisation (choice of the attribute and possibility to skip _pollution_ tokens
  regardless of the attribute)
- Matching can be performed on custom attributes more easily
- Qualifiers are regrouped together within the `edsnlp.qualifiers` submodule, the inheritance from the `GenericMatcher` is dropped.
- `edsnlp.utils.filter.filter_spans` now accepts a `label_to_remove` parameter. If set, only corresponding spans are removed, along with overlapping spans. Primary use-case: removing pseudo cues for
  qualifiers.
- Generalise the naming convention for extensions, which keep the same name as the pipeline that created them (eg `Span._.negation` for the `eds.negation` pipeline). The previous convention is kept
  for now, but calling it issues a warning.
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

## v0.3.2 (2021-11-24)

- Major revamp of the normalisation.
    - The `normalizer` pipeline **now adds atomic components** (`lowercase`, `accents`, `quotes`, `pollution` & `endlines`) to the processing pipeline, and compiles the results into a
      new `Doc._.normalized` extension. The latter is itself a spaCy `Doc` object, wherein tokens are normalised and pollution tokens are removed altogether. Components that match on the `CUSTOM_NORM`
      attribute process the `normalized` document, and matches are brought back to the original document using a token-wise mapping.
    - Update the `RegexMatcher` to use the `CUSTOM_NORM` attribute
    - Add an `EDSPhraseMatcher`, wrapping spaCy's `PhraseMatcher` to enable matching on `CUSTOM_NORM`.
    - Update the `matcher` and `advanced` pipelines to enable matching on the `CUSTOM_NORM` attribute.
- Add an OMOP connector, to help go back and forth between OMOP-formatted pandas dataframes and spaCy documents.
- Add a `reason` pipeline, that extracts the reason for visit.
- Add an `endlines` pipeline, that classifies newline characters between spaces and actual ends of line.
- Add possibility to annotate within entities for qualifiers (`negation`, `hypothesis`, etc), ie if the cue is within the entity. Disabled by default.

## v0.3.1 (2021-10-13)

- Update `dates` to remove miscellaneous bugs.
- Add `isort` pre-commit hook.
- Improve performance for `negation`, `hypothesis`, `antecedents`, `family` and `rspeech` by using spaCy's `filter_spans` and our `consume_spans` methods.
- Add proposition segmentation to `hypothesis` and `family`, enhancing results.

## v0.3.0 (2021-09-29)

- Renamed `generic` to `matcher`. This is a non-breaking change for the average user, adding the pipeline is still :

  ```{ .python .no-check }
  nlp.add_pipe("matcher", config=dict(terms=dict(maladie="maladie")))
  ```

- Removed `quickumls` pipeline. It was untested, unmaintained. Will be added back in a future release.
- Add `score` pipeline, and `charlson`.
- Add `advanced-regex` pipeline
- Corrected bugs in the `negation` pipeline

## v0.2.0 (2021-09-13)

- Add `negation` pipeline
- Add `family` pipeline
- Add `hypothesis` pipeline
- Add `antecedents` pipeline
- Add `rspeech` pipeline
- Refactor the library :
    - Remove the `rules` folder
    - Add a `pipelines` folder, containing one subdirectory per component
    - Every component subdirectory contains a module defining the component, and a module defining a factory, plus any other utilities (eg `terms.py`)

## v0.1.0 (2021-09-29)

First working version. Available pipelines :

- `section`
- `sentences`
- `normalization`
- `pollution`
