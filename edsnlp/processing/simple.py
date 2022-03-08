from typing import Any, Dict, List, Union

import pandas as pd
import spacy
from spacy import Language
from spacy.tokens import Doc, Span
from tqdm import tqdm

nlp = spacy.blank("fr")

ExtensionSchema = Union[
    str,
    List[str],
    Dict[str, Any],
]


def _df_to_spacy(
    note: pd.DataFrame,
):
    """
    Takes a pandas DataFrame and return a generator that can be used in
    `nlp.pipe()`.

    Parameters
    ----------
    note: pd.DataFrame
        A pandas DataFrame with at least `note_text` and `note_id` columns.
        A `Doc` object will be created for each line.

    Returns
    -------
    generator:
        A generator which items are of the form (text, context), with `text`
        being a string and `context` a dictionnary
    """

    for col in ["note_text", "note_id"]:
        if col not in note.columns:
            raise ValueError(f"No column named {repr(col)} found in df")

    kept_note = note[["note_text", "note_id"]]
    if not Doc.has_extension("note_id"):
        Doc.set_extension("note_id", default=None)

    yield from zip(kept_note.note_text, kept_note.note_id)


def _flatten(list_of_lists: List[List[Any]]):
    """
    Flatten a list of lists to a combined list.
    """
    return [item for sublist in list_of_lists for item in sublist]


def _pipe_generator(
    note: pd.DataFrame,
    nlp: Language,
    additional_spans: Union[List[str], str] = "discarded",
    extensions: ExtensionSchema = [],
    batch_size: int = 1000,
    progress_bar: bool = True,
):

    if type(extensions) == str:
        extensions = [extensions]
    elif type(extensions) == dict:
        extensions = extensions.keys()

    if type(additional_spans) == str:
        additional_spans = [additional_spans]

    gen = _df_to_spacy(note)
    n_docs = len(note)
    pipeline = nlp.pipe(gen, as_tuples=True, batch_size=batch_size)

    for doc, note_id in tqdm(pipeline, total=n_docs, disable=not progress_bar):
        doc._.set("note_id", note_id)

        yield _full_schema(
            doc,
            additional_spans=additional_spans,
            extensions=extensions,
        )


def _single_schema(
    ent: Span,
    span_type: str = "ents",
    extensions: List[str] = [],
):

    return {
        "note_id": ent.doc._.note_id,
        "lexical_variant": ent.text,
        "label": ent.label_,
        "span_type": span_type,
        "start": ent.start_char,
        "end": ent.end_char,
        **{extension: getattr(ent._, extension) for extension in extensions},
    }


def _full_schema(
    doc: Doc,
    additional_spans: List[str] = [],
    extensions: List[str] = [],
):
    """
    Function used when Parallelising tasks via joblib.
    Takes a Doc as input, and returns a list of serializable objects

    !!! note

        The parallelisation needs for output objects to be **serializable**:
        after splitting the task into separate jobs, intermediate results
        are saved on memory before being aggregated, thus the need to be
        serializable. For instance, spaCy's spans aren't serializable since
        they are merely a *view* of the parent document.

        Check the source code of this function for an example.

    """

    results = []

    results.extend(
        [
            _single_schema(
                ent,
                extensions=extensions,
            )
            for ent in doc.ents
            if doc.ents
        ]
    )

    for span_type in additional_spans:
        results.extend(
            [
                _single_schema(
                    ent,
                    span_type=span_type,
                    extensions=extensions,
                )
                for ent in doc.spans[span_type]
                if doc.spans[span_type]
            ]
        )
    return results


def pipe(
    note: pd.DataFrame,
    nlp: Language,
    additional_spans: Union[List[str], str] = "discarded",
    extensions: Union[List[str], str] = [],
    batch_size: int = 1000,
    progress_bar: bool = True,
):
    """
    Function to apply a spaCy pipe to a pandas DataFrame note
    For a large DataFrame, prefer the parallel version.

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
        For instance, if `extensions=["score_name"]`, the extracted result
        will include, for each entity, `ent._.score_name`.
    batch_size : int, by default 1000
        Batch size used by spaCy's pipe
    progress_bar: bool, by default True
        Whether to display a progress bar or not

    Returns
    -------
    DataFrame
        A pandas DataFrame with one line per extraction
    """
    return pd.DataFrame(
        _flatten(
            _pipe_generator(
                note=note,
                nlp=nlp,
                additional_spans=additional_spans,
                extensions=extensions,
                batch_size=batch_size,
                progress_bar=progress_bar,
            )
        )
    )
