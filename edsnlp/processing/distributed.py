from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

from decorator import decorator
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from spacy import Language

from edsnlp.pipelines.base import BaseComponent
from edsnlp.processing.typing import DataFrameModules, DataFrames, get_module


def pyspark_type_finder(obj):
    """
    Returns (when possible) the PySpark type of any python object
    """
    try:
        inferred_type = T._infer_type(obj)
        print(f"Inferred type is {repr(inferred_type)}")
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
    additional_spans: Union[List[str], str] = "discarded",
    extensions: List[Tuple[str, T.DataType]] = [],
) -> DataFrame:
    """
    Function to apply a spaCy pipe to a pyspark or koalas DataFrame note

    Parameters
    ----------
    note : DataFrame
        A Pyspark or Koalas DataFrame with a `note_id` and `note_text` column
    nlp : Language
        A spaCy pipe
    additional_spans : Union[List[str], str], by default "discarded"
        A name (or list of names) of SpanGroup on which to apply the pipe too:
        SpanGroup are available as `doc.spans[spangroup_name]` and can be generated
        by some pipes. For instance, the `date` pipe populates doc.spans['dates']
    extensions : List[Tuple[str, T.DataType]], by default []
        Spans extensions to add to the extracted results:
        FOr instance, if `extensions=["score_name"]`, the extracted result
        will include, for each entity, `ent._.score_name`.

    Returns
    -------
    DataFrame
        A pyspark DataFrame with one line per extraction
    """
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    sc = spark.sparkContext

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
                        T.StructField(extension_name, extension_type, True)
                        for extension_name, extension_type in extensions.items()
                    ],
                ]
            )
        )

        def f(
            text,
            additional_spans=additional_spans,
            extensions=extensions,
        ):

            if text is None:
                return []

            nlp = nlp_bc.value

            for _, pipe in nlp.pipeline:
                if isinstance(pipe, BaseComponent):
                    pipe.set_extensions()

            doc = nlp(text)

            ents = []

            for ent in doc.ents:
                parsed_extensions = [
                    getattr(ent._, extension) for extension in extensions.keys()
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
                        getattr(ent._, extension) for extension in extensions.keys()
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

    note_nlp = note.withColumn("matches", matcher(note.note_text))
    note_nlp = note_nlp.withColumn("matches", F.explode(note_nlp.matches))

    note_nlp = note_nlp.select("note_id", "matches.*")

    return note_nlp
