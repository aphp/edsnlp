# Inference

Once you have obtained a pipeline, either by composing rule-based components, training a model or loading a model from the disk, you can use it to make predictions on documents. This is referred to as inference. This page answers the following questions :

> How do we leverage computational resources run a model on many documents?

> How do we connect to various data sources to retrieve documents?

Be sure to check out the [Processing multiple texts](/tutorials/multiple-texts) tutorial for a practical example of how to use EDS-NLP to process large datasets.

## Inference on a single document

In EDS-NLP, computing the prediction on a single document is done by calling the pipeline on the document. The input can be either:

- a text string
- or a [Doc](https://spacy.io/api/doc) object

```{ .python .no-check }
from pathlib import Path

nlp = ...
text = "... my text ..."
doc = nlp(text)
```

If you're lucky enough to have a GPU, you can use it to speed up inference by moving the model to the GPU before calling the pipeline.

```{ .python .no-check }
nlp.to("cuda")  # same semantics as pytorch
doc = nlp(text)
```

To leverage multiple GPUs when processing multiple documents, refer to the [multiprocessing backend][edsnlp.processing.multiprocessing.execute_multiprocessing_backend] description below.

## Streams

When processing multiple documents, we can optimize the inference by parallelizing the computation on a single core, multiple cores and GPUs or even multiple machines.

These optimizations are enabled by performing *lazy inference* : the operations (e.g., reading a document, converting it to a Doc, running the different pipes of a model or writing the result somewhere) are not executed immediately but are instead scheduled in a [Stream][edsnlp.core.stream.Stream] object. It can then be executed by calling the `execute` method, iterating over it or calling a writing method (e.g., `to_pandas`). In fact, data connectors like `edsnlp.data.read_json` return a stream, as well as the `nlp.pipe` method.

A stream contains :

- a `reader`: the source of the data (e.g., a file, a database, a list of strings, etc.)
- the list of operations to perform (`stream.ops`) that contain the function / pipe, keyword arguments and context for each operation
- an optional `writer`: the destination of the data (e.g., a file, a database, a list of strings, etc.)
- the execution `config`, containing the backend to use and its configuration such as the number of workers, the batch size, etc.

All methods (`map()`, `map_batches()`, `map_gpu()`, `map_pipeline()`, `set_processing()`) of the stream are chainable, meaning that they return a new stream object (no in-place modification).

For instance, the following code will load a model, read a folder of JSON files, apply the model to each document and write the result in a Parquet folder, using 4 CPUs and 2 GPUs.

```{ .python .no-check }
import edsnlp

# Load or create a model
nlp = edsnlp.load("path/to/model")

# Read some data (this is lazy, no data will be read until the end of of this snippet)
data = edsnlp.data.read_json("path/to/json_folder", converter="...")

# Apply each pipe of the model to our documents and split the data
# into batches such that each contains at most 100 000 padded words
# (padded_words = max doc size in batch * batch size)
data = data.map_pipeline(
    nlp,
    # optional arguments
    batch_size=100_000,
    batch_by="padded_words",
)
# or equivalently : data = nlp.pipe(data, batch_size=100_000, batch_by="padded_words")

# Sort the documents in chunks of 1024 documents
data = data.map_batches(
    lambda batch: sorted(batch, key=lambda doc: len(doc)),
    batch_size=1024,
)

data = data.map_batches(
    # Apply a function to each batch of documents
    lambda batch: [doc._.my_custom_attribute for doc in batch]
)

# Configure the execution
data = data.set_processing(
    # 4 CPUs to parallelize rule-based pipes, IO and preprocessing
    num_cpu_workers=4,
    # 2 GPUs to accelerate deep-learning pipes
    num_gpu_workers=2,
    # Show the progress bar
    show_progress=True,
)

# Write the result, this will execute the stream
data.write_parquet("path/to/output_folder", converter="...", write_in_worker=True)
```

Streams support a variety of operations, such as applying a function to each element of the stream, batching the elements, applying a model to the elements, etc. In each case, the operations will not be executed immediately but will be scheduled to be executed when iterating of the collection, or calling the `execute()`, `to_*()` or `write_*()` methods.

### `map()` {: #edsnlp.core.stream.Stream.map }

::: edsnlp.core.stream.Stream.map
    options:
        sections: ['text', 'parameters']
        header: false
        show_source: false

### `map_batches()` {: #edsnlp.core.stream.Stream.map_batches }

To apply an operation to a stream in batches, you can use the `map_batches()` method. It takes a callable as input, an optional dictionary of keyword arguments and batching arguments.

::: edsnlp.core.stream.Stream.map_batches
    options:
        heading_level: 3
        sections: ['text', 'parameters']
        header: false
        show_source: false

### `map_pipeline()` {: #edsnlp.core.stream.Stream.map_pipeline }

::: edsnlp.core.stream.Stream.map_pipeline
    options:
        heading_level: 3
        sections: ['text', 'parameters']
        header: false
        show_source: false

### `map_gpu()` {: #edsnlp.core.stream.Stream.map_gpu }

::: edsnlp.core.stream.Stream.map_gpu
    options:
        heading_level: 3
        sections: ['text', 'parameters']
        header: false
        show_source: false

### `loop()` {: #edsnlp.core.stream.Stream.loop }

::: edsnlp.core.stream.Stream.loop
    options:
        heading_level: 3
        sections: ['text', 'parameters']
        header: false
        show_source: false

### `shuffle()` {: #edsnlp.core.stream.Stream.shuffle }

::: edsnlp.core.stream.Stream.shuffle
    options:
        heading_level: 3
        sections: ['text', 'parameters']
        header: false
        show_source: false

### Configure the execution with `set_processing()` {: #edsnlp.core.stream.Stream.set_processing }

You can configure how the operations performed in the stream is executed by calling its `set_processing(...)` method. The following options are available :

::: edsnlp.core.stream.Stream.set_processing
    options:
        heading_level: 3
        sections: ['text', 'parameters']
        header: false
        show_source: false

## Backends

The `backend` parameter of the `set_processing` supports the following values:

### `simple` {: #edsnlp.processing.simple.execute_simple_backend }

::: edsnlp.processing.simple.execute_simple_backend
    options:
        heading_level: 3
        show_source: false

### `multiprocessing` {: #edsnlp.processing.multiprocessing.execute_multiprocessing_backend }

::: edsnlp.processing.multiprocessing.execute_multiprocessing_backend
    options:
        heading_level: 3
        show_source: false

### `spark` {: #edsnlp.processing.spark.execute_spark_backend }

::: edsnlp.processing.spark.execute_spark_backend
    options:
        heading_level: 3
        show_source: false

## Batching

Many operations rely on batching, either to be more efficient or because they require a fixed-size input. The `batch_size` and `batch_by` argument of the `map_batches()` method allows you to specify the size of the batches and what function to use to compute the size of the batches.

```{ .python .no-check }
# Accumulate in chunks of 1024 documents
lengths = data.map_batches(len, batch_size=1024)

# Accumulate in chunks of 100 000 words
lengths = data.map_batches(len, batch_size=100_000, batch_by="words")
# or
lengths = data.map_batches(len, batch_size="100_000 words")
```

We also support special values for `batch_size` which use "sentinels" (i.e. markers inserted in the stream) to delimit the batches.

```{ .python .no-check }
# Accumulate every element of the input in a single batch
# which is useful when looping over the data in training
lengths = data.map_batches(len, batch_size="dataset")

# Accumulate in chunks of fragments, in the case of parquet datasets
lengths = data.map_batches(len, batch_size="fragments")
```

Note that these batch functions are only available under specific conditions:

- either `backend="simple"` or `deterministic=True` (default) if `backend="multiprocessing"`, otherwise elements might be processed out of order
- if every op before was elementwise (e.g. `map()`, `map_gpu()`, `map_pipeline()` and no generator function), or `sentinel_mode` was explicitly set to `"split"` in `map_batches()`, otherwise the sentinel are dropped by default when the user requires batching.
