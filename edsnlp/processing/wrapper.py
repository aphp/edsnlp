from typing import List, Union

import pandas as pd
import pyspark.sql as ps
from spacy import Language

from .parallel import pipe as parallel_pipe
from .simple import extentions_schema
from .simple import pipe as simple_pipe
from .spark import pipe as spark_pipe


def pipe(
    note: Union[pd.DataFrame, ps.DataFrame],
    nlp: Language,
    how: str = "parallel",
    additional_spans: Union[List[str], str] = "discarded",
    extensions: extentions_schema = [],
    **kwargs,
) -> Union[pd.DataFrame, ps.DataFrame]:
    """
    Function to apply a SpaCy pipe to a pandas or pyspark DataFrame


    Parameters
    ----------
    note : DataFrame
        A pandas DataFrame with a `note_id` and `note_text` column
    nlp : Language
        A SpaCy pipe
    how : str, by default "parallel"
        3 methods are avaiable here:

        - `how='simple'`: Single process on a Pandas DataFrame
        - `how='parallel'`: Parallelized processes on a Pandas DataFrame
        - `how='spark'`: Distributed processes on a pyspark DataFrame
    additional_spans : Union[List[str], str], by default "discarded"

        A name (or list of names) of SpanGroup on which to apply the pipe too:
        SpanGroup are available as `doc.spans[spangroup_name]` and can be generated
        by some pipes. For instance, the `date` pipe populates doc.spans['dates']
    extensions : List[Tuple[str, T.DataType]], by default []
        Spans extensions to add to the extracted results:
        For instance, if `extensions=["score_name"]`, the extracted result
        will include, for each entity, `ent._.score_name`.
    kwargs:
        Additionnal parameters depending on the `how` argument.

    Returns
    -------
    Union[pd.DataFrame, ps.DataFrame]
        A DataFrame with one line per extraction
    """

    if (type(note) == ps.DataFrame) and (how != "spark"):
        raise ValueError(
            "You are providing a pyspark DataFrame, please use `how='spark'`"
        )
    if how == "simple":

        return simple_pipe(
            note=note,
            nlp=nlp,
            additional_spans=additional_spans,
            extensions=extensions,
            **kwargs,
        )

    if how == "parallel":

        return parallel_pipe(
            note=note,
            nlp=nlp,
            additional_spans=additional_spans,
            extensions=extensions,
            **kwargs,
        )

    if how == "spark":
        if type(note) == pd.DataFrame:
            raise ValueError(
                """
                You are providing a pandas DataFrame with `how='spark'`,
                which is incompatible.
                """
            )

        if extensions and type(extensions) != dict:
            raise ValueError(
                """
                When using Spark, you should provide extension names
                along with the extension type (as a dictionnary):
                `d[extension_name] = extension_type`
                """  # noqa W291
            )

        return spark_pipe(
            note=note,
            nlp=nlp,
            additional_spans=additional_spans,
            extensions=extensions,
            **kwargs,
        )
