# Processing multiple texts

In the previous tutorials, we've seen how to apply a SpaCy NLP pipeline to a single text. Once the pipeline is tested and ready to be applied on an entire corpus, we'll want to deploy it efficiently.

In this tutorial, we'll cover a few best practices and some _caveats_ to avoid. Then, we'll explore methods that EDS-NLP provides to use a SpaCy pipeline directly on a Pandas or Spark DataFrame. These can drastically increase throughput (up to 20x speed increase on our 64-core machines).

Consider this simple pipeline:

```python title="Pipeline definition"
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

It turns out SpaCy has a powerful parallelisation engine for an efficient processing of multiple texts.
So the first step for writing more efficient SpaCy code is to use `nlp.pipe` when processing multiple texts:

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
parallelizing the processing can be really efficient (up to 20x faster), but will require a (tiny) bit more work.
Here are shown 4 ways to analyse texts depending on your needs

A [wrapper](#wrapper) is available to simply switch between those use cases.

## Processing a pandas DataFrame

Processing text within a pandas DataFrame is a very common use case. In many applications, you'll select a corpus of documents over a distributed cluster, load it in memory and process all texts.

To make sure we can follow along, we propose two recipes for getting the DataFrame: through Spark or using a dummy dataset like before.

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

=== "Using a Spark DataFrame as input"

    ```python
    from pyspark.sql.session import SparkSession

    spark = SparkSession.builder.getOrCreate()

    df = spark.sql("SELECT * FROM note")
    df = df.select("note_id", "note_text")

    data = df.limit(1000).toPandas()
    ```

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

The result on the first note:

| note_id | start | end | label      | lexical_variant   | negation | hypothesis | family | key   |
| ------: | ----: | --: | :--------- | :---------------- | -------: | ---------: | -----: | :---- |
|       0 |     0 |   7 | patient    | Patient           |        0 |          0 |      0 | ents  |
|       0 |   114 | 121 | patient    | patient           |        0 |          0 |      1 | ents  |
|       0 |    17 |  34 | 2021-09-25 | 25 septembre 2021 |      nan |        nan |    nan | dates |

## Using EDS-NLP's helper functions

Let's see how we can efficiently deploy our pipeline using EDS-NLP's utility methods.

They share the same arguments:

| Argument           | Description                                                     | Default  |
| ------------------ | --------------------------------------------------------------- | -------- |
| `note`             | A DataFrame, with two required columns, `note_id` and `note_id` | Required |
| `nlp`              | The pipeline object                                             | Required |
| `additional_spans` | Keys in `doc.spans` to include besides `doc.ents`               | `[]`     |
| `extensions`       | Custom extensions to use                                        | `[]`     |

Depending on your pipeline, you may want ot extract other extensions. To do so, simply provide those extension names (without the leading underscore) to the `extensions` argument.

### Single process

EDS-NLP provides a [`simple_pipe` helper][edsnlp.processing.simple.pipe] function that avoids the hassle we just went through in the previous section. Using it is trivial:

```python
# ↑ Omitted code above ↑
from edsnlp.processing import single_pipe

note_nlp = simple_pipe(
    data,
    nlp,
    additional_spans=["dates"],
    extensions=["parsed_date"],
)
```

In just two Python statements, we get the exact same result as before!

### Multiple processes

Depending on the size of your corpus, and if you have CPU cores to spare, you may want to distribute the computation. Again, EDS-NLP makes it extremely easy for you, through the [`parallel_pipe` helper][edsnlp.processing.parallel.pipe]:

```python
# ↑ Omitted code above ↑
from edsnlp.processing import parallel_pipe

note_nlp = parallel_pipe(
    data,
    nlp,
    additional_spans=["dates"],
    extensions=["parsed_date"],
    n_jobs=-2,  # (1)
)
```

1. The `n_jobs` parameter controls the number of workers that you deploy in parallel. Negative inputs means "all cores minus `#!python abs(n_jobs)`"

!!! danger "Using a large number of workers and memory use"

    In SpaCy, even a rule-based pipeline is a memory intensive object.
    Be wary of using too many workers, lest you get a memory error.

Depending on your machine, you should get a significant speed boost (we got 20x acceleration).

## Deploying EDS-NLP on Spark

Should you need to deploy SpaCy on larger-than-memory Spark DataFrames, EDS-NLP has you covered.

There is a minor twist, though: Spark needs to know in advance the type of each extension you want to save. Thus, if you need additional extensions to be saved, you'll have to provide a dictionary to the `extensions` argument instead of a list of strings. This dictionary will have the name of the extension as keys and its PySpark type as value.

Accepted types are the ones present in [`pyspark.sql.types`](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql.html#data-types){ target="\_blank"}.

EDS-NLP provides a helper function, [`pyspark_type_finder`][edsnlp.processing.spark.pyspark_type_finder], is available to get the correct type for most Python objects. You just need to provide an example of the type you wish to collect:

```python
dt_type = pyspark_type_finder(datetime.datetime(2020, 1, 1))
```

!!! danger "Be careful when providing the example"

    **Do not blindly provide the first entity matched by your pipeline**: it might be ill-suited. For instance, the `Span._.date` makes sense for a date span,
    but will be `None` if you use an entity...

Once again, using the helper is trivial:

```python
# ↑ Omitted code above ↑
from edsnlp.processing import spark_pipe

note_nlp = spark_pipe(
    df,  # (1)
    nlp,
    additional_spans=["dates"],
    extensions={"parsed_date": dt_type},
)

# Check that the pipeline was correctly distributed:
note_nlp.show(5)
```

1. We called the Spark DataFrame `df` in the earlier example.

Using Spark, you can deploy EDS-NLP pipelines on tens of millions of documents with ease!

## One function to rule them all

EDS-NLP provides a wrapper to simplify deployment even further:

```python
# ↑ Omitted code above ↑
from edsnlp.processing import pipe

### Small pandas DataFrame
note_nlp = pipe(
    note=df.limit(1000).toPandas(),
    nlp=nlp,
    how="simple",
    additional_spans=["dates"],
    extensions=["parsed_date"],
)

### Larger pandas DataFrame
note_nlp = pipe(
    note=df.limit(10000).toPandas(),
    nlp=nlp,
    how="parallel",
    additional_spans=["dates"],
    extensions=["parsed_date"],
)

### Huge Spark DataFrame
note_nlp = pipe(
    note=df,
    nlp=nlp,
    how="spark",
    additional_spans=["dates"],
    extensions={"parsed_date": dt_type},
)
```
