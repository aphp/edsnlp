from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import pandas as pd
from confit import VisibleDeprecationWarning
from spacy.tokens import Doc

import edsnlp.data

from ..core import PipelineProtocol
from ..utils.span_getters import SpanGetterArg, validate_span_setter
from ..utils.spark_dtypes import PySparkPrettyPrinter

if TYPE_CHECKING:
    from databricks.koalas.frame import DataFrame as KoalasDataFrame
    from pyspark.sql.dataframe import DataFrame as SparkDataFrame

ExtensionSchema = Union[
    str,
    List[str],
    Dict[str, Any],
]


def slugify(chained_attr: str) -> str:
    """
    Slugify a chained attribute name

    Parameters
    ----------
    chained_attr : str
        The string to slugify (replace dots by _)

    Returns
    -------
    str
        The slugified string
    """
    return chained_attr.replace(".", "_")


def pipe(
    df: Union[SparkDataFrame, KoalasDataFrame, pd.DataFrame],
    nlp: PipelineProtocol,
    n_jobs: int = -2,
    # Legacy parameters
    context: List[str] = [],
    results_extractor: Optional[Callable[[Doc], List[Dict[str, Any]]]] = None,
    additional_spans: SpanGetterArg = [],
    extensions: ExtensionSchema = [],
    dtypes: Any = None,
    **kwargs,
) -> Union[SparkDataFrame, KoalasDataFrame, pd.DataFrame]:
    """
    Helper to process a pandas, koalas or spark dataframe. This function is deprecated.
    Prefer using the following instead:

    ```{ .python .no-check }
    import edsnlp

    docs = edsnlp.data.from_***(
        df,
        converter='omop',
        doc_attributes=context,
    )
    docs = docs.map_pipeline(nlp)
    res = edsnlp.data.to_***(
        docs,
        converter='ents',  # or custom extractor
        span_getter="ents",
        span_attributes=span_attributes,
        **kwargs
    )
    ```

    You can also call this function to get a migration suggestion.

    Parameters
    ----------
    df: Union[SparkDataFrame, KoalasDataFrame, pd.DataFrame]
        The dataframe to process, can be a pandas, spark or koalas dataframe
    nlp: PipelineProtocol
        The pipeline to use
    n_jobs: int
        Number of CPU workers to use
    context: List[str]
        List of context attributes to keep
    results_extractor: Optional[Callable[[Doc], List[Dict[str, Any]]]]
        Function to extract results from the pipeline. Defaults to one row per
        entities.
    additional_spans: SpanGetterArg
        Additional spans groups to keep, defaults to `ents` (doc.ents)
    extensions: ExtensionSchema
        Span extensions to export as a column. Can be a list of extension names, a dict
        of extension names to types, or a string
    dtypes: Any
        Spark schema to use for the output dataframe. This is only used if
        the input dataframe is a spark dataframe.
    kwargs: Any
        Additional keyword arguments to pass to the `edsnlp.data.to_*` function

    Returns
    -------
    Union[SparkDataFrame, KoalasDataFrame, pd.DataFrame]
        The processed dataframe
    """
    is_pandas = isinstance(df, pd.DataFrame)
    s = (
        "edsnlp.processing.pipe is deprecated, use the following instead:\n\n"
        "import edsnlp\n"
        f"docs = edsnlp.data.from_{'pandas' if is_pandas else 'spark'}(\n"
        f"    df,\n"
        f"    converter='omop',\n"
        f"    doc_attributes={context!r}\n"
        f")\n"
        f"docs = docs.map_pipeline(nlp)\n"
        f"res = edsnlp.data.to_{'pandas' if is_pandas else 'spark'}(\n"
        f"    docs,\n"
        f"    converter={repr('ents') if results_extractor is None else 'extractor'},\n"
    )
    if not results_extractor and additional_spans:
        s += f"    span_getter={repr('ents' if additional_spans is None else [*additional_spans, 'ents'])},\n"  # noqa: E501
    if not results_extractor and extensions:
        s += f"    span_attributes={repr({ext: slugify(ext) for ext in extensions})},\n"
    if dtypes:
        s += f"    dtypes={PySparkPrettyPrinter().pformat(dtypes)},\n"
    s += ")"
    warnings.warn(s, VisibleDeprecationWarning)

    write_span_getter = validate_span_setter(
        "ents" if additional_spans is None else [*additional_spans, "ents"],
    )

    if isinstance(df, pd.DataFrame):
        docs = edsnlp.data.from_pandas(
            df,
            converter="omop",
            doc_attributes=context,
        )
        docs = docs.map_pipeline(nlp)
        docs = docs.set_processing(num_cpu_workers=n_jobs)
        return (
            edsnlp.data.to_pandas(
                docs,
                converter="ents",
                span_getter=write_span_getter,
                span_attributes={ext: slugify(ext) for ext in extensions},
                **kwargs,
            )
            if results_extractor is None
            else edsnlp.data.to_pandas(
                docs,
                converter=results_extractor,
            )
        )
    import pyspark.sql.types as T

    try:
        KoalasDataFrame = sys.modules["databricks.koalas.frame"].DataFrame
    except (AttributeError, KeyError):
        KoalasDataFrame = None
    is_koalas = KoalasDataFrame and isinstance(df, KoalasDataFrame)  # type: ignore
    if is_koalas:
        df: SparkDataFrame = df.to_spark()  # type: ignore

    docs = edsnlp.data.from_spark(
        df,
        converter="omop",
        doc_attributes=context,
    )
    docs = docs.map_pipeline(nlp)

    schema = None

    if results_extractor is None:
        if dtypes is None and (not extensions or isinstance(extensions, dict)):
            schema = [
                df.schema["note_id"],
                T.StructField("lexical_variant", T.StringType(), False),
                T.StructField("label", T.StringType(), False),
                T.StructField("start", T.IntegerType(), False),
                T.StructField("end", T.IntegerType(), False),
                T.StructField("span_type", T.StringType(), False),
            ]

        if isinstance(extensions, dict):
            schema = [
                *schema,
                *(
                    T.StructField(slugify(extension_name), extension_type, True)
                    for extension_name, extension_type in extensions.items()
                ),
            ]
        converter = "ents"
    else:
        if dtypes:
            schema = [
                df.schema["note_id"],
            ] + [
                T.StructField(slugify(extension_name), extension_type, True)
                for extension_name, extension_type in dtypes.items()
            ]

        def converter(doc):
            res = results_extractor(doc)
            return (
                [{"note_id": doc._.note_id, **row} for row in res]
                if isinstance(res, list)
                else {"note_id": doc._.note_id, **res}
            )

    schema = T.StructType(schema) if schema is not None else None

    docs = (
        edsnlp.data.to_spark(
            docs,
            converter="ents",
            span_getter=write_span_getter,
            span_attributes={ext: slugify(ext) for ext in extensions},
            dtypes=schema,
        )
        if results_extractor is None
        else edsnlp.data.to_spark(
            docs,
            converter=converter,
            dtypes=schema,
        )
    )

    if is_koalas:
        docs = docs.to_koalas()

    return docs
