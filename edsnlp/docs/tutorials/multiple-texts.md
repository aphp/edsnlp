# Processing multiple texts

In the previous tutorials, we've seen how to apply a spaCy NLP pipeline to a single text.
Once the pipeline is tested and ready to be applied on an entire corpus, we'll want to deploy it efficiently.

In this tutorial, we'll cover a few best practices and some _caveats_ to avoid.
Then, we'll explore methods that EDS-NLP provides to use a spaCy pipeline directly on a pandas or Spark DataFrame.
These can drastically increase throughput.

Consider this simple pipeline:

```python title="Pipeline definition: pipeline.py"
import spacy

nlp = spacy.blank("fr")

nlp.add_pipe("eds.sentences")
nlp.add_pipe("eds.normalizer")

config = dict(
    terms=dict(patient=["patient", "malade"]),
    attr="NORM",
)
nlp.add_pipe("eds.matcher", config=config)

# Add qualifiers
nlp.add_pipe("eds.negation")
nlp.add_pipe("eds.hypothesis")
nlp.add_pipe("eds.family")

# Add date detection
nlp.add_pipe("eds.dates")
```

Let's deploy it on a large number of documents.

## What about a `for` loop?

Suppose we have a corpus of text:

```python
text = (
    "Patient admis le 25 septembre 2021 pour suspicion de Covid.\n"
    "Pas de cas de coronavirus dans ce service.\n"
    "Le père du patient est atteint du covid."
)

corpus = [text] * 10000  # (1)
```

1. This is admittedly ugly. But you get the idea, we have a corpus of 10 000 documents we want to process...

You _could_ just apply the pipeline document by document.

```python title="A naive approach"
# ↑ Omitted code above ↑

docs = [nlp(text) for text in corpus]
```

It turns out spaCy has a powerful parallelisation engine for an efficient processing of multiple texts.
So the first step for writing more efficient spaCy code is to use `nlp.pipe` when processing multiple texts:

```diff
- docs = [nlp(text) for text in corpus]
+ docs = list(nlp.pipe(corpus))
```

The `nlp.pipe` method takes an iterable as input, and outputs a generator of `Doc` object. Under the hood, texts are processed in batches, which is often much more efficient.

!!! info "Batch processing and EDS-NLP"

    For now, EDS-NLP does not natively parallelise its components, so the gain from using `nlp.pipe` will not be that significant.

    Nevertheless, it's good practice to avoid using `for` loops when possible. Moreover, you will benefit from the batched tokenisation step.

The way EDS-NLP is used may depend on how many documents you are working with.
Once working with tens of thousands of them,
parallelising the processing can be really efficient (up to 20x faster), but will require a (tiny) bit more work.
Here are shown 4 ways to analyse texts depending on your needs

A [wrapper](#one-function-to-rule-them-all) is available to simply switch between those use cases.

## Processing a pandas DataFrame

Processing text within a pandas DataFrame is a very common use case. In many applications, you'll select a corpus of documents over a distributed cluster, load it in memory and process all texts.

!!! note "The OMOP CDM"

    In every tutorial that mentions distributing EDS-NLP over a corpus of documents,
    we will expect the data to be organised using a flavour of the
    [OMOP Common Data Model](https://ohdsi.github.io/CommonDataModel/).

    The OMOP CDM defines two tables of interest to us:

    - the [`note` table](https://ohdsi.github.io/CommonDataModel/cdm54.html#NOTE) contains the clinical notes
    - the [`note_nlp` table](https://ohdsi.github.io/CommonDataModel/cdm54.html#NOTE_NLP) holds the results of
      a NLP pipeline applied to the `note` table.

To make sure we can follow along, we propose three recipes for getting the DataFrame: using a dummy dataset like before, loading a CSV or by loading a Spark DataFrame into memory.

=== "Dummy example"

    ```python
    import pandas as pd

    text = (
        "Patient admis le 25 septembre 2021 pour suspicion de Covid.\n"
        "Pas de cas de coronavirus dans ce service.\n"
        "Le père du patient est atteint du covid."
    )

    corpus = [text] * 1000

    data = pd.DataFrame(dict(note_text=corpus))
    data["note_id"] = range(len(data))
    ```

=== "Loading data from a CSV"

    <!-- no-check -->

    ```python
    import pandas as pd

    data = pd.read_csv("note.csv")
    ```

=== "Loading data from a Spark DataFrame"

    <!-- no-check -->

    ```python
    from pyspark.sql.session import SparkSession

    spark = SparkSession.builder.getOrCreate()

    df = spark.sql("SELECT * FROM note")
    df = df.select("note_id", "note_text")

    data = df.limit(1000).toPandas()  # (1)
    ```

    1. We limit the size of the DataFrame to make sure we do not overwhelm our machine.

We'll see in what follows how we can efficiently deploy our pipeline on the `#!python data` object.

## "By hand"

We can deploy the pipeline using `nlp.pipe` directly, but we'll need some work to format the results in a usable way. Let's see how this might go, before using EDS-NLP's helper function to avoid the boilerplate code.

```python title="processing.py"
from spacy.tokens import Doc
from typing import Any, Dict, List


def get_entities(doc: Doc) -> List[Dict[str, Any]]:
    """Return a list of dict representation for the entities"""

    entities = []

    for ent in doc.ents:
        d = dict(
            start=ent.start_char,
            end=ent.end_char,
            label=ent.label_,
            lexical_variant=ent.text,
            negation=ent._.negation,
            hypothesis=ent._.hypothesis,
            family=ent._.family,
            key="ents",
        )
        entities.append(d)

    for date in doc.spans.get("dates", []):
        d = dict(
            start=date.start_char,
            end=date.end_char,
            label=date._.date,
            lexical_variant=date.text,
            key="dates",
        )
        entities.append(d)

    return entities
```

<!-- no-check -->

```python
# ↑ Omitted code above ↑
from processing import get_entities
import pandas as pd

data["doc"] = list(nlp.pipe(data.note_text))  # (1)
data["entities"] = data.doc.apply(get_entities)  # (2)

# "Explode" the dataframe
data = data[["note_id", "entities"]].explode("entities")
data = data.dropna()

data = data.reset_index(drop=True)

data = data[["note_id"]].join(pd.json_normalize(data.entities))
```

1. We use spaCy's efficient `nlp.pipe` method
2. This part is far from optimal, since it uses apply... But the computationally heavy part is in the previous line,
   since `get_entities` merely _reads_ pre-computed values from the document.

The result on the first note:

| note_id | start |  end | label      | lexical_variant   | negation | hypothesis | family | key   |
| ------: | ----: | ---: | :--------- | :---------------- | -------: | ---------: | -----: | :---- |
|       0 |     0 |    7 | patient    | Patient           |        0 |          0 |      0 | ents  |
|       0 |   114 |  121 | patient    | patient           |        0 |          0 |      1 | ents  |
|       0 |    17 |   34 | 2021-09-25 | 25 septembre 2021 |      nan |        nan |    nan | dates |

## Using EDS-NLP's helper functions

Let's see how we can efficiently deploy our pipeline using EDS-NLP's utility methods.

They share the same arguments:

| Argument            | Description                                                                   | Default                 |
| ------------------- | ----------------------------------------------------------------------------- | ----------------------- |
| `note`              | A DataFrame, with two required columns, `note_id` and `note_text`             | Required                |
| `nlp`               | The pipeline object                                                           | Required                |
| `context`           | A list of column names to add context to the generate `Doc`                   | `[]`                    |
| `additional_spans`  | Keys in `doc.spans` to include besides `doc.ents`                             | `[]`                    |
| `extensions`        | Custom extensions to use                                                      | `[]`                    |
| `results_extractor` | An arbitrary callback function that turns a `Doc` into a list of dictionaries | `None` (use extensions) |

!!! tip "Adding context"

    You might want to store some context information contained in the `note` DataFrame as an extension in the generated `Doc` object.

    For instance, you may use the [`eds.dates` pipeline](../pipelines/misc/dates.md)
    in coordination with the `note_datetime` field to normalise a relative date
    (eg `Le patient est venu il y a trois jours/The patient came three days ago`).

    In this case, you can use the `context` parameter and provide a list of column names you want to add:

    <!-- no-check -->

    ```python
    note_nlp = single_pipe(
        data,
        nlp,
        context=["note_datetime"],
        additional_spans=["dates"],
        extensions=["date.day", "date.month", "date.year"],
    )
    ```

    In this example, the `note_datetime` field becomes available as `doc._.note_datetime`.

Depending on your pipeline, you may want to extract other extensions.
To do so, simply provide those extension names (without the leading underscore) to the `extensions` argument.
**This should cover most use-cases**.

In case you need more fine-grained control over how you want to process the results of your pipeline,
you can provide an arbitrary `results_extractor` function. Said function is expected to take a spaCy `Doc`
object as input, and return a list of dictionaries that will be used to construct the `note_nlp` table.
For instance, the `get_entities` function defined earlier could be distributed directly:

<!-- no-check -->

```python
# ↑ Omitted code above ↑
from edsnlp.processing.simple import pipe as single_pipe
from processing import get_entities

note_nlp = single_pipe(
    data,
    nlp,
    results_extractor=get_entities,
)
```

!!! danger "A few caveats on using an arbitrary function"

    Should you use multiprocessing, your arbitrary function needs to be serialisable
    as a pickle object in order to be distributed. That implies a few limitations on the way your
    function can be defined.

    Namely, your **function needs to be discoverable** (see the [pickle documentation on the subject](https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled)). When deploying it should be defined such a way that can be accessed by the worker processes.

    For that reason, **arbitrary functions can only be distributed via Spark/Koalas if their source code is advertised to the Spark workers**.
    To that end, you should define your custom function in a pip-installed Python package.

### Single process

EDS-NLP provides a [`single_pipe`][edsnlp.processing.simple.pipe] helper function that avoids the hassle we just went through in the previous section. Using it is trivial:

```python
# ↑ Omitted code above ↑
from edsnlp.processing.simple import pipe as single_pipe

note_nlp = single_pipe(
    data,
    nlp,
    additional_spans=["dates"],
    extensions=["date.day", "date.month", "date.year"],
)
```

In just two Python statements, we get the exact same result as before!

### Multiple processes

Depending on the size of your corpus, and if you have CPU cores to spare, you may want to distribute the computation. Again, EDS-NLP makes it extremely easy for you, through the [`parallel_pipe`][edsnlp.processing.parallel.pipe] helper:

```python
# ↑ Omitted code above ↑
from edsnlp.processing.parallel import pipe as parallel_pipe

note_nlp = parallel_pipe(
    data,
    nlp,
    additional_spans=["dates"],
    extensions=["date.day", "date.month", "date.year"],
    n_jobs=-2,  # (1)
)
```

1. The `n_jobs` parameter controls the number of workers that you deploy in parallel. Negative inputs means "all cores minus `#!python abs(n_jobs + 1)`"

!!! danger "Using a large number of workers and memory use"

    In spaCy, even a rule-based pipeline is a memory intensive object.
    Be wary of using too many workers, lest you get a memory error.

Depending on your machine, you should get a significant speed boost (we got 20x acceleration on a shared cluster using 62 cores).

## Deploying EDS-NLP on Spark/Koalas

Should you need to deploy spaCy on a distributed DataFrame such as a [Spark](https://spark.apache.org/) or a [Koalas](https://koalas.readthedocs.io/en/latest/index.html) DataFrame, EDS-NLP has you covered.
The procedure for those two types of DataFrame is virtually the same. Under the hood, EDS-NLP automatically deals with the necessary conversions.

Suppose you have a Spark DataFrame:

=== "Using a dummy example"

    ```python
    from pyspark.sql.session import SparkSession
    from pyspark.sql import types as T

    spark = SparkSession.builder.getOrCreate()

    schema = T.StructType(
        [
            T.StructField("note_id", T.IntegerType()),
            T.StructField("note_text", T.StringType()),
        ]
    )

    text = (
        "Patient admis le 25 septembre 2021 pour suspicion de Covid.\n"
        "Pas de cas de coronavirus dans ce service.\n"
        "Le père du patient est atteint du covid."
    )

    data = [(i, text) for i in range(1000)]

    df = spark.createDataFrame(data=data, schema=schema)
    ```

=== "Loading a pre-existing table"

    <!-- no-check -->

    ```python
    from pyspark.sql.session import SparkSession

    spark = SparkSession.builder.getOrCreate()

    df = spark.sql("SELECT * FROM note")
    df = df.select("note_id", "note_text")
    ```

=== "Using a Koalas DataFrame"

    <!-- no-check -->

    ```python
    from pyspark.sql.session import SparkSession
    import databricks.koalas

    spark = SparkSession.builder.getOrCreate()

    df = spark.sql("SELECT note_id, note_text FROM note").to_koalas()
    ```

### Declaring types

There is a minor twist, though: Spark (or Koalas) needs to know in advance the type of each extension you want to save. Thus, if you need additional extensions to be saved, you'll have to provide a dictionary to the `extensions` argument instead of a list of strings. This dictionary will have the name of the extension as keys and its PySpark type as value.

Accepted types are the ones present in [`pyspark.sql.types`](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql.html#data-types){ target="\_blank"}.

EDS-NLP provides a helper function, [`pyspark_type_finder`][edsnlp.processing.distributed.pyspark_type_finder], is available to get the correct type for most Python objects. You just need to provide an example of the type you wish to collect:

<!-- no-check -->

```python
int_type = pyspark_type_finder(1)

# Out: IntegerType()
```

!!! danger "Be careful when providing the example"

    **Do not blindly provide the first entity matched by your pipeline**: it might be ill-suited. For instance, the `Span._.date` makes sense for a date span,
    but will be `None` if you use an entity...

### Deploying the pipeline

Once again, using the helper is trivial:

=== "Spark"

    <!-- no-check -->

    ```python
    # ↑ Omitted code above ↑
    from edsnlp.processing.distributed import pipe as distributed_pipe

    note_nlp = distributed_pipe(
        df,
        nlp,
        additional_spans=["dates"],
        extensions={"date.year": int_type, "date.month": int_type, "date.day": int_type},
    )

    # Check that the pipeline was correctly distributed:
    note_nlp.show(5)
    ```

=== "Koalas"

    <!-- no-check -->

    ```python
    # ↑ Omitted code above ↑
    from edsnlp.processing.distributed import pipe as distributed_pipe

    note_nlp = distributed_pipe(
        df,
        nlp,
        additional_spans=["dates"],
        extensions={"date.year": int_type, "date.month": int_type, "date.day": int_type},
    )

    # Check that the pipeline was correctly distributed:
    note_nlp.head()
    ```

Using Spark or Koalas, you can deploy EDS-NLP pipelines on tens of millions of documents with ease!

## One function to rule them all

EDS-NLP provides a wrapper to simplify deployment even further:

<!-- no-check -->

```python
# ↑ Omitted code above ↑
from edsnlp.processing import pipe

### Small pandas DataFrame
note_nlp = pipe(
    note=df.limit(1000).toPandas(),
    nlp=nlp,
    n_jobs=1,
    additional_spans=["dates"],
    extensions=["date.day", "date.month", "date.year"],
)

### Larger pandas DataFrame
note_nlp = pipe(
    note=df.limit(10000).toPandas(),
    nlp=nlp,
    n_jobs=-2,
    additional_spans=["dates"],
    extensions=["date.day", "date.month", "date.year"],
)

### Huge Spark or Koalas DataFrame
note_nlp = pipe(
    note=df,
    nlp=nlp,
    how="spark",
    additional_spans=["dates"],
    extensions={"date.year": int_type, "date.month": int_type, "date.day": int_type},
)
```
