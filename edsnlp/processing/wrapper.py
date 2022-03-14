from typing import Any, Dict, List, Union

from spacy import Language

from .parallel import pipe as parallel_pipe
from .simple import ExtensionSchema
from .simple import pipe as simple_pipe
from .typing import DataFrameModules, DataFrames, get_module


def pipe(
    note: DataFrames,
    nlp: Language,
    n_jobs: int = -2,
    additional_spans: Union[List[str], str] = "discarded",
    extensions: ExtensionSchema = [],
    **kwargs: Dict[str, Any],
) -> DataFrames:
    """
    Function to apply a spaCy pipe to a pandas or pyspark DataFrame


    Parameters
    ----------
    note : DataFrame
        A pandas/pyspark/koalas DataFrame with a `note_id` and `note_text` column
    nlp : Language
        A spaCy pipe
    n_jobs : int, by default -2
        Only used when providing a Pandas DataFrame

        - `n_jobs=1` corresponds to `simple_pipe`
        - `n_jobs>1` corresponds to `parallel_pipe` with `n_jobs` parallel workers
        - `n_jobs=-1` corresponds to `parallel_pipe` with maximun number of workers
        - `n_jobs=-2` corresponds to `parallel_pipe` with maximun number of workers -1
    additional_spans : Union[List[str], str], by default "discarded"
        A name (or list of names) of SpanGroup on which to apply the pipe too:
        SpanGroup are available as `doc.spans[spangroup_name]` and can be generated
        by some pipes. For instance, the `date` pipe populates doc.spans['dates']
    extensions : List[Tuple[str, T.DataType]], by default []
        Spans extensions to add to the extracted results:
        For instance, if `extensions=["score_name"]`, the extracted result
        will include, for each entity, `ent._.score_name`.
    kwargs : Dict[str, Any]
        Additional parameters depending on the `how` argument.

    Returns
    -------
    DataFrame
        A DataFrame with one line per extraction
    """

    module = get_module(note)

    if module == DataFrameModules.PANDAS:
        if n_jobs == 1:

            return simple_pipe(
                note=note,
                nlp=nlp,
                additional_spans=additional_spans,
                extensions=extensions,
                **kwargs,
            )

        else:

            return parallel_pipe(
                note=note,
                nlp=nlp,
                additional_spans=additional_spans,
                extensions=extensions,
                n_jobs=n_jobs,
                **kwargs,
            )

    if extensions and type(extensions) != dict:
        raise ValueError(
            """
            When using Spark or Koalas, you should provide extension names
            along with the extension type (as a dictionnary):
            `d[extension_name] = extension_type`
            """  # noqa W291
        )

    from .distributed import pipe as distributed_pipe

    return distributed_pipe(
        note=note,
        nlp=nlp,
        additional_spans=additional_spans,
        extensions=extensions,
        **kwargs,
    )
