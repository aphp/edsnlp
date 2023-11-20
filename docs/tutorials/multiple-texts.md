# Processing multiple texts

In the previous tutorials, we've seen how to apply a spaCy NLP pipeline to a single text.
Once the pipeline is tested and ready to be applied on an entire corpus, we'll want to deploy it efficiently.

In this tutorial, we'll cover a few best practices and some _caveats_ to avoid.
Then, we'll explore methods that EDS-NLP provides perform inference on multiple texts.

Consider this simple pipeline:

```python
import edsnlp

nlp = edsnlp.blank("eds")

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

```python
# ↑ Omitted code above ↑

docs = [nlp(text) for text in corpus]
```

Next, you might want to convert these documents to a DataFrame for further analysis:

```python
import edsnlp.data

df = edsnlp.data.to_pandas(docs, converter="omop")
```

There are a few issues with this approach:

- If our model contains deep learning components (which it does not in this tutorial), we don't benefit from optimized batched matrix operations : ideally, we'd like to process multiple documents at
  once.
- We may have multiple cores available but we don't use them to apply the pipes of our model to multiple documents at the same time.
- We would also like to perform the conversion step (`converter="omop"` which extracts the annotations of our Doc object into dictionaries) in parallel.

## Lazy inference and parallelization

To efficiently perform the same operations on multiple documents at once, EDS-NLP uses [lazy collections][edsnlp.core.lazy_collection.LazyCollection], which record the operations to perform on the documents without actually executing them directly. This allows EDS-NLP to distribute these operations on multiple cores or machines when it is time to execute them. We can configure how the collection operations are run (how many jobs/workers, how many gpus, whether to use the spark engine) via the lazy collection [`.set_processing(...)`][edsnlp.core.lazy_collection.LazyCollection.set_processing] method.

For instance,

```python
docs = edsnlp.data.from_iterable(corpus)
```

as well as any `edsnlp.data.read_*` or `edsnlp.data.from_*` return a lazy collection, that we can iterate over or complete with more operations. To apply the model on our collection of documents, we can simply do:

```python
docs = docs.map_pipeline(nlp)
# or à la spaCy :
# docs = nlp.pipe(docs)
```

!!! warning "SpaCy vs EDS-NLP"

    SpaCy's `nlp.pipe` method is not the same as EDS-NLP's `nlp.pipe` method, and will iterate over anything you pass to it, therefore executing the operations scheduled in our lazy collection.

    We recommend you instantiate your models using `nlp = edsnlp.blank(...)` or `nlp = edsnlp.load(...)`.

    Otherwise, use the following to apply a spaCy model on a lazy collection `docs` without triggering its execution:

    ```{ .python .no-check }
    docs = docs.map_pipeline(nlp)
    ```

Finally, we can convert the documents to a DataFrame (or other formats / files) using the `edsnlp.data.to_*` or `edsnlp.data.write_*` methods. This triggers the execution of the operations scheduled in the lazy collection and produces the rows of the DataFrame.

```python
df = docs.to_pandas(converter="omop")
# or equivalently:
# df = edsnlp.data.to_pandas(docs, converter="omop")
```

We can also iterate over the documents, which also triggers the execution of the operations scheduled in the lazy collection.

```python
for doc in docs:
    # do something with the doc
    pass
```

## Processing a DataFrame

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

    ```{ .python .no-check }
    import pandas as pd

    data = pd.read_csv("note.csv")
    ```

=== "Loading data from a Spark DataFrame"

    ```{ .python .no-check }
    from pyspark.sql.session import SparkSession

    spark = SparkSession.builder.getOrCreate()

    df = spark.sql("SELECT * FROM note")
    df = df.select("note_id", "note_text")

    data = df.limit(1000).toPandas()  # (1)
    ```

    1. We limit the size of the DataFrame to make sure we do not overwhelm our machine.

We'll see in what follows how we can efficiently deploy our pipeline on the `#!python data` object.

### Locally without parallelization

```python
# Read from a dataframe & use the omop converter
docs = edsnlp.data.from_pandas(data, converter="omop")

# Add the pipeline to operations that will be run
docs = nlp.pipe(docs)

# Convert each doc to a list of dicts (one by entity)
# and store the result in a pandas DataFrame
note_nlp = edsnlp.data.to_pandas(
    docs,
    converter="ents",
    # Below are the arguments to the converter
    span_getter=["ents", "dates"],
    span_attributes={  # (1)
        "negation": "negation",
        "hypothesis": "hypothesis",
        "family": "family",
        "date.day": "date_day",  # slugified name
        "date.month": "date_month",
        "date.year": "date_year",
    },
)
```

1. You can just pass a list if you don't want to rename the attributes.

The result on the first note:

| note_id | start | end | label      | lexical_variant   | negation | hypothesis | family | key   |
|--------:|------:|----:|:-----------|:------------------|---------:|-----------:|-------:|:------|
|       0 |     0 |   7 | patient    | Patient           |        0 |          0 |      0 | ents  |
|       0 |   114 | 121 | patient    | patient           |        0 |          0 |      1 | ents  |
|       0 |    17 |  34 | 2021-09-25 | 25 septembre 2021 |      nan |        nan |    nan | dates |

### Locally, using multiple parallel workers

```{ .python hl_lines="8" }
# Read from a dataframe & use the omop converter
docs = edsnlp.data.from_pandas(data, converter="omop")

# Add the pipeline to operations that will be run
docs = nlp.pipe(docs)

# The operations of our lazy collection will be distributed on multiple workers
docs = docs.set_processing(backend="multiprocessing")

# Convert each doc to a list of dicts (one by entity)
# and store the result in a pandas DataFrame
note_nlp = edsnlp.data.to_pandas(
    docs,
    converter="ents",
    span_getter=["ents", "dates"],
    span_attributes={
        "negation": "negation",
        "hypothesis": "hypothesis",
        "family": "family",
        "date.day": "date_day",  # slugify the extension name
        "date.month": "date_month",
        "date.year": "date_year"
    },
)
```

### In a distributed fashion with spark

To use the Spark engine to distribute the computation, we create our lazy collection from the Spark dataframe directly and write the result to a new Spark dataframe. EDS-NLP will automatically distribute the operations on the cluster (setting `backend="spark"` behind the scenes), but you can change the backend (for instance to `multiprocessing` to run locally).

```{ .python hl_lines="2 9" .no-check }
# Read from the pyspark dataframe & use the omop converter
docs = edsnlp.data.from_spark(df, converter="omop")

# Add the pipeline to operations that will be run
docs = nlp.pipe(docs)

# Convert each doc to a list of dicts (one by entity)
# and store the result in a pyspark DataFrame
note_nlp = edsnlp.data.to_spark(
    docs,
    converter="ents",
    span_getter=["ents", "dates"],
    span_attributes={
        "negation": "negation",
        "hypothesis": "hypothesis",
        "family": "family",
        "date.day": "date_day",  # slugify the extension name
        "date.month": "date_month",
        "date.year": "date_year"
    },
    dtypes=None,  # (1)
)
```

1. If you don't pass a `dtypes` argument, EDS-NLP will print the inferred schema it such that you can copy-paste it in your code.

### Using a custom converter

To customize the conversion of a Doc object to dictionaries, you can pass a `converter` argument. It will either be a string (the name of a converter) or a callable, that should return either a dictionary or a list of dictionaries.

```python
from spacy.tokens import Doc
from typing import Any, Dict, List


def get_entities(doc: Doc) -> List[Dict[str, Any]]:
    """Return a list of dict representation for the entities"""

    entities = []

    for ent in doc.ents:
        d = dict(
            begin=ent.start_char,
            end=ent.end_char,
            label=ent.label_,
            entity_text=ent.text,
            negation=ent._.negation,
            hypothesis=ent._.hypothesis,
            family=ent._.family,
        )
        entities.append(d)

    for date in doc.spans.get("dates", []):
        d = dict(
            begin=date.start_char,
            end=date.end_char,
            label="date",
            entity_text=date.text,
        )
        entities.append(d)

    return entities


docs = edsnlp.data.from_pandas(data, converter="omop")

# Add the pipeline to operations that will be run
docs = nlp.pipe(docs)

# Convert each doc to a list of dicts (one by entity)
# and store the result in a pyspark DataFrame
note_nlp = edsnlp.data.to_pandas(
    docs,
    converter=get_entities,
    # no keyword args here since our converter expects none
)
```

| begin | end |   label | entity_text | negation | hypothesis | family |
|------:|----:|--------:|------------:|---------:|-----------:|-------:|
|     0 |   7 | patient |     Patient |    False |      False |  False |
|   114 | 121 | patient |     patient |    False |      False |   True |
|    17 |  34 |    date |  25 sept... |          |            |        |
|     0 |   7 | patient |     Patient |    False |      False |  False |
|   114 | 121 | patient |     patient |    False |      False |   True |
