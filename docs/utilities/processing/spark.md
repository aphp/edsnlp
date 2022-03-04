# Deploying on Spark

We provide a simple connector to distribute a pipeline on a Spark cluster. We expose a Spark UDF (user-defined function) factory that handles the nitty gritty of distributing a pipeline over a cluster of Spark-enabled machines.

## Distributing a pipeline

Because of the way Spark distributes Python objects, we need to re-declare custom extensions on the executors. To make this step as smooth as possible, EDS-NLP provides a `BaseComponent` class that implements a `set_extensions` method. When the pipeline is distributed, every component that extend `BaseComponent` rerun their `set_extensions` method.

Since SpaCy `Doc` objects cannot easily be serialised, the UDF we provide returns a list of detected entities along with selected qualifiers.

## Example

See the [dedicated tutorial](../../home/tutorials/multiple-texts.md) for a step-by-step presentation.

## Authors and citation

The Spark connector was developed by AP-HP's Data Science team.
