from typing import Iterable, List, Union

import pandas as pd
import spacy
from joblib import Parallel, delayed
from spacy import Language

from .simple import ExtensionSchema, _flatten, _pipe_generator

nlp = spacy.blank("fr")


def _define_nlp(new_nlp: Language):
    """
    Set the global nlp variable
    Doing it this way saves non negligeable amount of time
    """
    global nlp
    nlp = new_nlp


def _chunker(
    iterable: Iterable,
    total_length: int,
    chunksize: int,
):
    """
    Takes an iterable and chunk it.
    """
    return (
        iterable[pos : pos + chunksize] for pos in range(0, total_length, chunksize)
    )


def _process_chunk(note: pd.DataFrame, **pipe_kwargs):

    list_results = []

    for out in _pipe_generator(note, nlp, progress_bar=False, **pipe_kwargs):
        list_results += out

    return list_results


def pipe(
    note: pd.DataFrame,
    nlp: Language,
    additional_spans: Union[List[str], str] = "discarded",
    extensions: ExtensionSchema = [],
    chunksize: int = 100,
    n_jobs: int = -2,
    progress_bar: bool = True,
    **pipe_kwargs,
):
    """
    Function to apply a spaCy pipe to a pandas DataFrame note by using multiprocessing

    Parameters
    ----------
    note : DataFrame
        A pandas DataFrame with a `note_id` and `note_text` column
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
    chunksize: int, by default 100
        Batch size used to split tasks
    n_jobs: int, by default -2
        Max number of parallel jobs.
        The default value uses the maximum number of available cores.
    progress_bar: bool, by default True
        Whether to display a progress bar or not
    **pipe_kwargs:
        Arguments exposed in `processing.pipe_generator` are also available here

    Returns
    -------
    DataFrame
        A pandas DataFrame with one line per extraction
    """

    # Setting the nlp variable
    _define_nlp(nlp)

    verbose = 10 if progress_bar else 0

    executor = Parallel(
        n_jobs, backend="multiprocessing", prefer="processes", verbose=verbose
    )
    executor.warn(f"Used nlp components: {nlp.component_names}")

    pipe_kwargs["additional_spans"] = additional_spans
    pipe_kwargs["extensions"] = extensions

    if verbose:
        executor.warn(f"{int(len(note)/chunksize)} tasks to complete")

    do = delayed(_process_chunk)

    tasks = (
        do(chunk, **pipe_kwargs)
        for chunk in _chunker(note, len(note), chunksize=chunksize)
    )
    result = executor(tasks)

    out = _flatten(result)

    return pd.DataFrame(out)
