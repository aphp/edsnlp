# Data connectors

We provide various connectors to read and write data from and to different formats.

Reading from a given path or object takes the following form:

```{ .python .no-check }
import edsnlp

docs = edsnlp.data.read_{format}(  # or .from_{format} for objects
    # Path to the file or directory
    "path/to/file",
    # How to convert JSON-like samples to Doc objects
    converter=predefined schema or function,
)
```

Writing to given path or object takes the following form:

```{ .python .no-check }
import edsnlp

edsnlp.data.write_{format}(  # or .to_{format} for objects
    # Path to the file or directory
    "path/to/file",
    # Iterable of Doc objects
    docs,
    # How to convert Doc objects to JSON-like samples
    converter=predefined schema or function,
)
```

The overall process is illustrated in the following diagram:

![Data connectors overview](./overview.png)

At the moment, we support the following data sources:

| Source                        | Description                |
|:------------------------------|:---------------------------|
| [JSON](./json)                | `.json` and `.jsonl` files |
| [Standoff & BRAT](./standoff) | `.ann` and `.txt` files    |
| [Pandas](./pandas)            | Pandas DataFrame objects   |
| [Polars](./polars)            | Polars DataFrame objects   |
| [Spark](./spark)              | Spark DataFrame objects    |

and the following schemas:

| Schema                                                              | Snippet                |
|:--------------------------------------------------------------------|------------------------|
| [Custom](./converters/#custom)                                      | `converter=custom_fn`  |
| [OMOP](./converters/#omop)                                          | `converter="omop"`     |
| [Standoff](./converters/#standoff)                                  | `converter="standoff"` |
| [Ents](./converters/#edsnlp.data.converters.EntsDoc2DictConverter)  | `converter="ents"`     |
| [Markup](./converters/#edsnlp.data.converters.MarkupToDocConverter) | `converter="markup"`   |
