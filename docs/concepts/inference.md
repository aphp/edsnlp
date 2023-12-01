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

## Inference on multiple documents {: #edsnlp.core.lazy_collection.LazyCollection }

When processing multiple documents, we can optimize the inference by parallelizing the computation on a single core, multiple cores and GPUs or even multiple machines.

### Lazy collection

These optimizations are enabled by performing *lazy inference* : the operations (e.g., reading a document, converting it to a Doc, running the different pipes of a model or writing the result somewhere) are not executed immediately but are instead scheduled in a [LazyCollection][edsnlp.core.lazy_collection.LazyCollection] object. It can then be executed by calling the `execute` method, iterating over it or calling a writing method (e.g., `to_pandas`). In fact, data connectors like `edsnlp.data.read_json` return a lazy collection, as well as the `nlp.pipe` method.

A lazy collection contains :

- a `reader`: the source of the data (e.g., a file, a database, a list of strings, etc.)
- the list of operations to perform under a `pipeline` attribute containing the name if any, function / pipe, keyword arguments and context for each operation
- an optional `writer`: the destination of the data (e.g., a file, a database, a list of strings, etc.)
- the execution `config`, containing the backend to use and its configuration such as the number of workers, the batch size, etc.

All methods (`.map`, `.map_pipeline`, `.set_processing`) of the lazy collection are chainable, meaning that they return a new object (no in-place modification).

For instance, the following code will load a model, read a folder of JSON files, apply the model to each document and write the result in a Parquet folder, using 4 CPUs and 2 GPUs.

```{ .python .no-check }
import edsnlp

# Load or create a model
nlp = edsnlp.load("path/to/model")

# Read some data (this is lazy, no data will be read until the end of of this snippet)
data = edsnlp.data.read_json("path/to/json_folder", converter="...")

# Apply each pipe of the model to our documents
data = data.map_pipeline(nlp)
# or equivalently : data = nlp.pipe(data)

# Configure the execution
data = data.set_processing(
    # 4 CPUs to parallelize rule-based pipes, IO and preprocessing
    num_cpu_workers=4,
    # 2 GPUs to accelerate deep-learning pipes
    num_gpu_workers=2,
)

# Write the result, this will execute the lazy collection
data.write_parquet("path/to/output_folder", converter="...", write_in_worker=True)
```

### Applying operations to a lazy collection

To apply an operation to a lazy collection, you can use the `.map` method. It takes a callable as input and an optional dictionary of keyword arguments. The function will be applied to each element of the collection.

To apply a model, you can use the `.map_pipeline` method. It takes a model as input and will add every pipe of the model to the scheduled operations.

In both cases, the operations will not be executed immediately but will be scheduled to be executed when iterating of the collection, or calling the `.execute`, `.to_*` or `.write_*` methods.

### Execution of a lazy collection {: #edsnlp.core.lazy_collection.LazyCollection.set_processing }

You can configure how the operations performed in the lazy collection is executed by calling its `set_processing(...)` method. The following options are available :

::: edsnlp.core.lazy_collection.LazyCollection.set_processing
    options:
        heading_level: 3
        only_parameters: "no-header"

## Backends

### Simple backend {: #edsnlp.processing.simple.execute_simple_backend }

::: edsnlp.processing.simple.execute_simple_backend
    options:
        heading_level: 3
        show_source: false

### Multiprocessing backend {: #edsnlp.processing.multiprocessing.execute_multiprocessing_backend }

::: edsnlp.processing.multiprocessing.execute_multiprocessing_backend
    options:
        heading_level: 3
        show_source: false

### Spark backend {: #edsnlp.processing.spark.execute_spark_backend }

::: edsnlp.processing.spark.execute_spark_backend
    options:
        heading_level: 3
        show_source: false
