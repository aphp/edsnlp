from functools import partial
from typing import Any, Callable, Dict, List, Union

from decorator import decorator
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from spacy import Language
from spacy.tokens import Doc

from edsnlp.pipelines.base import BaseComponent
from edsnlp.utils.extensions import rgetattr

from .helpers import (
    DataFrameModules,
    DataFrames,
    check_spacy_version_for_context,
    get_module,
    slugify,
)


def pyspark_type_finder(obj):
    """
    Returns (when possible) the PySpark type of any python object
    """
    try:
        inferred_type = T._infer_type(obj)
        logger.info(f"Inferred type is {repr(inferred_type)}")
        return inferred_type
    except TypeError:
        raise TypeError("Cannot infer type for this object.")


@decorator
def module_checker(
    func: Callable,
    *args,
    **kwargs,
) -> Any:

    args = list(args)
    note = args.pop(0)
    module = get_module(note)

    if module == DataFrameModules.PYSPARK:
        return func(note, *args, **kwargs)
    elif module == DataFrameModules.KOALAS:
        import databricks.koalas  # noqa F401

        note_spark = note.to_spark()
        note_nlp_spark = func(note_spark, *args, **kwargs)
        return note_nlp_spark.to_koalas()


@module_checker
def pipe(
    note: DataFrames,
    nlp: Language,
    context: List[str] = [],
    additional_spans: Union[List[str], str] = "discarded",
    extensions: Dict[str, T.DataType] = {},
) -> DataFrame:
    """
    Function to apply a spaCy pipe to a pyspark or koalas DataFrame note

    Parameters
    ----------
    note : DataFrame
        A Pyspark or Koalas DataFrame with a `note_id` and `note_text` column
    nlp : Language
        A spaCy pipe
    context : List[str]
        A list of column to add to the generated SpaCy document as an extension.
        For instance, if `context=["note_datetime"], the corresponding value found
        in the `note_datetime` column will be stored in `doc._.note_datetime`,
        which can be useful e.g. for the `dates` pipeline.
    additional_spans : Union[List[str], str], by default "discarded"
        A name (or list of names) of SpanGroup on which to apply the pipe too:
        SpanGroup are available as `doc.spans[spangroup_name]` and can be generated
        by some pipes. For instance, the `eds.dates` pipeline
        component populates `doc.spans['dates']`
    extensions : List[Tuple[str, T.DataType]], by default []
        Spans extensions to add to the extracted results:
        For instance, if `extensions=["score_name"]`, the extracted result
        will include, for each entity, `ent._.score_name`.

    Returns
    -------
    DataFrame
        A pyspark DataFrame with one line per extraction
    """

    if context:
        check_spacy_version_for_context()

    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    sc = spark.sparkContext

    if not nlp.has_pipe("eds.context"):
        nlp.add_pipe("eds.context", first=True, config=dict(context=context))

    nlp_bc = sc.broadcast(nlp)

    def _udf_factory(
        additional_spans: Union[List[str], str] = "discarded",
        extensions: Dict[str, T.DataType] = dict(),
    ):

        schema = T.ArrayType(
            T.StructType(
                [
                    T.StructField("lexical_variant", T.StringType(), False),
                    T.StructField("label", T.StringType(), False),
                    T.StructField("span_type", T.StringType(), True),
                    T.StructField("start", T.IntegerType(), False),
                    T.StructField("end", T.IntegerType(), False),
                    *[
                        T.StructField(slugify(extension_name), extension_type, True)
                        for extension_name, extension_type in extensions.items()
                    ],
                ]
            )
        )

        def f(
            text,
            *context_values,
            additional_spans=additional_spans,
            extensions=extensions,
        ):

            if text is None:
                return []

            nlp = nlp_bc.value

            for _, pipe in nlp.pipeline:
                if isinstance(pipe, BaseComponent):
                    pipe.set_extensions()

            doc = nlp.make_doc(text)
            for context_name, context_value in zip(context, context_values):
                doc._.set(context_name, context_value)
            doc = nlp(doc)

            ents = []

            for ent in doc.ents:
                parsed_extensions = [
                    rgetattr(ent._, extension) for extension in extensions.keys()
                ]

                ents.append(
                    (
                        ent.text,
                        ent.label_,
                        "ents",
                        ent.start_char,
                        ent.end_char,
                        *parsed_extensions,
                    )
                )

            if additional_spans is None:
                return ents

            if type(additional_spans) == str:
                additional_spans = [additional_spans]

            for spans_name in additional_spans:

                for ent in doc.spans.get(spans_name, []):

                    parsed_extensions = [
                        rgetattr(ent._, extension) for extension in extensions.keys()
                    ]

                    ents.append(
                        (
                            ent.text,
                            ent.label_,
                            spans_name,
                            ent.start_char,
                            ent.end_char,
                            *parsed_extensions,
                        )
                    )

            return ents

        f_udf = F.udf(
            partial(
                f,
                additional_spans=additional_spans,
                extensions=extensions,
            ),
            schema,
        )

        return f_udf

    matcher = _udf_factory(
        additional_spans=additional_spans,
        extensions=extensions,
    )

    n_needed_partitions = max(note.count() // 2000, 1)  # Batch sizes of 2000

    note_nlp = note.repartition(n_needed_partitions).withColumn(
        "matches", matcher(F.col("note_text"), *[F.col(c) for c in context])
    )

    note_nlp = note_nlp.withColumn("matches", F.explode(note_nlp.matches))

    note_nlp = note_nlp.select("note_id", "matches.*")

    return note_nlp


@module_checker
def custom_pipe(
    note: DataFrames,
    nlp: Language,
    results_extractor: Callable[[Doc], List[Dict[str, Any]]],
    dtypes: Dict[str, T.DataType],
    context: List[str] = [],
) -> DataFrame:
    """
    Function to apply a spaCy pipe to a pyspark or koalas DataFrame note,
    a generic callback function that converts a spaCy `Doc` object into a
    list of dictionaries.

    Parameters
    ----------
    note : DataFrame
        A Pyspark or Koalas DataFrame with a `note_text` column
    nlp : Language
        A spaCy pipe
    results_extractor : Callable[[Doc], List[Dict[str, Any]]]
        Arbitrary function that takes extract serialisable results from the computed
        spaCy `Doc` object. The output of the function must be a list of dictionaries
        containing the extracted spans or entities.

        There is no requirement for all entities to provide every dictionary key.
    dtypes : Dict[str, T.DataType]
        Dictionary containing all expected keys from the `results_extractor` function,
        along with their types.
    context : List[str]
        A list of column to add to the generated SpaCy document as an extension.
        For instance, if `context=["note_datetime"], the corresponding value found
        in the `note_datetime` column will be stored in `doc._.note_datetime`,
        which can be useful e.g. for the `dates` pipeline.

    Returns
    -------
    DataFrame
        A pyspark DataFrame with one line per extraction
    """

    if context:
        check_spacy_version_for_context()

    if ("note_id" not in context) and ("note_id" in dtypes.keys()):
        context.append("note_id")

    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    sc = spark.sparkContext

    if not nlp.has_pipe("eds.context"):
        nlp.add_pipe("eds.context", first=True, config=dict(context=context))

    nlp_bc = sc.broadcast(nlp)

    schema = T.ArrayType(
        T.StructType([T.StructField(key, dtype) for key, dtype in dtypes.items()])
    )

    @F.udf(schema)
    def udf(
        text,
        *context_values,
    ):

        if text is None:
            return []

        nlp_ = nlp_bc.value

        for _, pipe in nlp.pipeline:
            if isinstance(pipe, BaseComponent):
                pipe.set_extensions()

        doc = nlp_.make_doc(text)
        for context_name, context_value in zip(context, context_values):
            doc._.set(context_name, context_value)

        doc = nlp_(doc)

        results = []

        for res in results_extractor(doc):
            results.append([res.get(key) for key in dtypes])

        return results

    note_nlp = note.withColumn(
        "matches", udf(F.col("note_text"), *[F.col(c) for c in context])
    )

    note_nlp = note_nlp.withColumn("matches", F.explode(note_nlp.matches))

    if ("note_id" not in dtypes.keys()) and ("note_id" in note_nlp.columns):
        note_nlp = note_nlp.select("note_id", "matches.*")
    else:
        note_nlp = note_nlp.select("matches.*")

    return note_nlp
