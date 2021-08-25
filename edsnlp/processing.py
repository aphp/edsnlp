from tqdm import tqdm
from typing import Tuple, List, Union, Any, Callable

import pandas as pd

import spacy
from spacy import Language
from spacy.tokens import Doc

from joblib import Parallel, delayed

nlp = spacy.blank("fr")


def df_to_spacy(
    df: pd.DataFrame,
    text_col: str = "note_text",
    context_cols: Union[str, List[str]] = [],
):
    """
    Takes a Pandas DataFrame and return a generator that can be used in
    `nlp.pipe()`

    Parameters
    ----------
    df:
        A Pandas DataFrame.
        A `Doc` object will be created for each line.
    text_col (str):
        The name of the column from `df` containing the to-be-analyzed text
    context_cols (str or list of str):
        column name or list of columns names of `df` containing attributes to add to the corresponding `Doc` object

    Returns
    -------
    generator:
        A generator which items are of the form (text, context), with `text` being a string and `context` a dictionnary
    """

    if type(context_cols) == str:
        context_cols = [context_cols]
    for col in [text_col] + context_cols:
        if col not in df.columns:
            raise ValueError(f"No column named {repr(col)} found in df")

    kept_df = df[[text_col] + context_cols]
    for c in context_cols:
        if not Doc.has_extension(c):
            Doc.set_extension(c, default=None)

    for d in kept_df.to_dict("records"):
        text = d.pop(text_col)
        yield (text, d)


def pipe(
    nlp: Language,
    df: pd.DataFrame,
    text_col: str = "note_text",
    context_cols: Union[str, List[str]] = [],
    batch_size: int = 1000,
    pick_results: Callable[[Doc], Any] = lambda x: x,
    progress_bar: bool = True,
):

    """
    Provides a generator based on the pipeline of the provided `nlp` object

    Parameters
    ----------
    nlp:
        The Spacy object.
    df:
        A Pandas DataFrame.
        A `Doc` object will be created for each line.
    text_col (str):
        The name of the column from `df` containing the to-be-analyzed text
    context_cols (str or list of str):
        column name or list of columns names of `df` containing attributes to add to the corresponding `Doc` object
    batch_size  (int):
        Batching size used for `nlp.pipe`
    pick_results (function, optionnal):
        Function applied to each `Doc` object before its yielded.
    progress_bar (bool):
        Whether to display a progress bar or not

    Return
    ------
    gen:
        A generator yielding a processed Doc object for each row from `df`
    """

    gen = df_to_spacy(df, text_col, context_cols)
    n_docs = len(df)
    pipeline = nlp.pipe(gen, as_tuples=True, batch_size=batch_size)

    for doc, context in tqdm(pipeline, total=n_docs, disable=not progress_bar):
        for k, v in context.items():
            doc._.set(k, v)
        yield pick_results(doc)


# Below are functions used for multiprocessing

from tqdm import tqdm


def define_nlp(new_nlp):
    """
    Set the global nlp variable
    Doing it this way saves non negligeable amount of time
    """
    global nlp
    nlp = new_nlp


def chunker(iterable, total_length, chunksize):
    """
    Takes an iterable and chunk it
    """
    return (
        iterable[pos : pos + chunksize] for pos in range(0, total_length, chunksize)
    )


def flatten(list_of_lists):
    """
    Flatten a list of lists to a combined list
    """
    return [item for sublist in list_of_lists for item in sublist]


def default_pick_results(doc):
    """
    Function used when Parallelizing tasks via joblib
    Takes a Doc as input, and returns a list of serializable objects
    """
    return [
        {
            "note_id": e.doc._.note_id if doc.has_extension("note_id") else None,
            "lexical_variant": e.text,
            "offset_start": e.start_char,
            "offset_end": e.end_char,
            "label": e.label_,
        }
        for e in doc.ents
        if doc.ents
    ]


def process_chunk(df, **pipe_kwargs):

    list_results = []

    for out in pipe(nlp, df, progress_bar=False, **pipe_kwargs):
        # Setting progress_bar at false because its comportment isn't satisfying during parallel jobs

        list_results += out

    return list_results


def parallel_pipe(
    nlp: Language,
    df: pd.DataFrame,
    chunksize: int = 100,
    n_jobs: int = 10,
    progress_bar: bool = True,
    **pipe_kwargs,
):
    """
    Wrapper to handle parallelisation of the provided nlp pipeline
    The method accepts the same parameters as the module.pipe() method, plus the additionnal `chunksize` and `n_jobs` params

    Parameters
    ----------
    nlp:
        The Spacy object.
    df:
        A Pandas DataFrame.
        A `Doc` object will be created for each line.
    text_col (str):
        The name of the column from `df` containing the to-be-analyzed text
    context_cols (str or list of str):
        column name or list of columns names of `df` containing attributes to add to the corresponding `Doc` object
    batch_size  (int):
        Batching size used for `nlp.pipe`
    pick_results (function, optionnal):
        Function applied to each `Doc` object before its yielded.
    progress_bar (bool):
        Whether to display a progress bar or not,
    chunksize (int):
        Batch size used to split tasks
    n_jobs (int):
        Max number of parallel jobs

    Return
    ------
    results (list):
        A list of outputs
        In pseudo-code, results is obtained as [pick_results(nlp(text)) for text in df.iterrows()]
    """

    # Setting the nlp variable
    define_nlp(nlp)

    verbose = 10 if progress_bar else 0

    executor = Parallel(
        n_jobs, backend="multiprocessing", prefer="processes", verbose=verbose
    )
    executor.warn(f"Used nlp components: {nlp.component_names}")

    if "pick_results" not in pipe_kwargs:
        pipe_kwargs["pick_results"] = default_pick_results

    if verbose:
        executor.warn(f"{int(len(df)/chunksize)} tasks to complete")

    do = delayed(process_chunk)

    tasks = (
        do(chunk, **pipe_kwargs) for chunk in chunker(df, len(df), chunksize=chunksize)
    )
    result = executor(tasks)

    return flatten(result)
