# ruff: noqa: F401
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import spacy.tokenizer
from fsspec import filesystem as fsspec
from loguru import logger
from typing_extensions import Literal

from edsnlp import registry
from edsnlp.core.stream import Stream
from edsnlp.data.base import BaseWriter, FileBasedReader
from edsnlp.data.converters import (
    FILENAME,
    AttributesMappingArg,
    get_dict2doc_converter,
    get_doc2dict_converter,
)
from edsnlp.utils.collections import flatten, shuffle
from edsnlp.utils.file_system import FileSystem, normalize_fs_path, walk_match
from edsnlp.utils.span_getters import SpanSetterArg
from edsnlp.utils.stream_sentinels import DatasetEndSentinel
from edsnlp.utils.typing import AsList

REGEX_ENTITY = re.compile(r"^(T\d+)\t(.*) (\d+ \d+(?:;\d+ \d+)*)\t(.*)$")
REGEX_NOTE = re.compile(r"^(#\d+)\tAnnotatorNotes ([^\t]+)\t(.*)$")
REGEX_RELATION = re.compile(r"^(R\d+)\t(\S+) Arg1:(\S+) Arg2:(\S+)")
REGEX_ATTRIBUTE = re.compile(r"^([AM]\d+)\t(.+?) ([TE]\d+)(?: (.+))?$")
REGEX_EVENT = re.compile(r"^(E\d+)\t(.+)$")
REGEX_EVENT_PART = re.compile(r"(\S+):([TE]\d+)")


class BratParsingError(ValueError):
    def __init__(self, ann_file, line):
        super().__init__(f"File {ann_file}, unrecognized Brat line {line}")


LOCAL_FS = fsspec("file")


def parse_standoff_file(
    txt_path: str,
    ann_paths: List[str],
    merge_spaced_fragments: bool = True,
    fs: FileSystem = LOCAL_FS,
) -> Dict:
    """
    Load a brat file

    Adapted from
    https://github.com/percevalw/nlstruct/blob/master/nlstruct/datasets/brat.py

    Parameters
    ----------
    path: str
        Path or glob path of the brat text file (.txt, not .ann)
    merge_spaced_fragments: bool
        Merge fragments of an entity that was split by brat because it overlapped an
        end of line
    fs: FileSystem
        Filesystem to use

    Returns
    -------
    Iterator[Dict]
    """
    entities = {}
    relations = []
    events = {}

    with fs.open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not len(ann_paths):
        return {
            "text": text,
        }

    for ann_file in ann_paths:
        with fs.open(ann_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                try:
                    if line.startswith("T"):
                        match = REGEX_ENTITY.match(line)
                        if match is None:
                            raise BratParsingError(ann_file, line)
                        ann_id = match.group(1)
                        entity = match.group(2)
                        span = match.group(3)
                        mention_text = match.group(4)
                        entities[ann_id] = {
                            "text": mention_text,
                            "entity_id": ann_id,
                            "fragments": [],
                            "attributes": {},
                            "notes": [],
                            "label": entity,
                        }
                        last_end = None
                        fragment_i = 0
                        begins_ends = sorted(
                            [
                                (int(s.split()[0]), int(s.split()[1]))
                                for s in span.split(";")
                            ]
                        )

                        for begin, end in begins_ends:
                            # If merge_spaced_fragments, merge two fragments that are
                            # only separated by a newline (brat automatically creates
                            # multiple fragments for a entity that spans over more than
                            # one line)
                            if (
                                merge_spaced_fragments
                                and last_end is not None
                                and len(text[last_end:begin].strip()) == 0
                            ):
                                entities[ann_id]["fragments"][-1]["end"] = end
                                last_end = end
                                continue
                            entities[ann_id]["fragments"].append(
                                {
                                    "begin": begin,
                                    "end": end,
                                }
                            )
                            fragment_i += 1
                            last_end = end
                    elif line.startswith("A") or line.startswith("M"):
                        match = REGEX_ATTRIBUTE.match(line)
                        if match is None:
                            raise BratParsingError(ann_file, line)
                        _, attr_name, entity_id, value = match.groups()
                        if attr_name is None:
                            raise BratParsingError(ann_file, line)
                        (
                            entities[entity_id]
                            if entity_id.startswith("T")
                            else events[entity_id]
                        )["attributes"][attr_name] = value
                    elif line.startswith("R"):
                        match = REGEX_RELATION.match(line)
                        if match is None:
                            raise BratParsingError(ann_file, line)
                        ann_id = match.group(1)
                        ann_name = match.group(2)
                        arg1 = match.group(3)
                        arg2 = match.group(4)
                        relations.append(
                            {
                                "relation_id": ann_id,
                                "relation_label": ann_name,
                                "from_entity_id": arg1,
                                "to_entity_id": arg2,
                            }
                        )
                    elif line.startswith("E"):
                        match = REGEX_EVENT.match(line)
                        if match is None:
                            raise BratParsingError(ann_file, line)
                        ann_id = match.group(1)
                        arguments_txt = match.group(2)
                        arguments = []
                        for argument in REGEX_EVENT_PART.finditer(arguments_txt):
                            arguments.append(
                                {
                                    "entity_id": argument.group(2),
                                    "label": argument.group(1),
                                }
                            )
                        events[ann_id] = {
                            "event_id": ann_id,
                            "attributes": {},
                            "arguments": arguments,
                        }
                    elif line.startswith("#"):
                        match = REGEX_NOTE.match(line)
                        if match is None:
                            raise BratParsingError(ann_file, line)
                        ann_id = match.group(1)
                        entity_id = match.group(2)
                        note = match.group(3)
                        entities[entity_id]["notes"].append(
                            {
                                "note_id": ann_id,
                                "value": note,
                            }
                        )
                except Exception:
                    raise Exception(
                        "Could not parse line {} from {}: {}".format(
                            line_idx, ann_file, repr(line)
                        )
                    )
    return {
        "text": text,
        "entities": list(entities.values()),
        "relations": relations,
        "events": list(events.values()),
    }


def dump_standoff_file(
    doc,
    txt_filename,
    overwrite_txt=False,
    overwrite_ann=False,
    fs: FileSystem = LOCAL_FS,
):
    parent_dir = txt_filename.rsplit("/", 1)[0]
    if parent_dir and not fs.exists(parent_dir):
        fs.makedirs(parent_dir, exist_ok=True)
    if not fs.exists(txt_filename) or overwrite_txt:
        with fs.open(txt_filename, "w", encoding="utf-8") as f:
            f.write(doc["text"])

    ann_filename = txt_filename.replace(".txt", ".ann")
    attribute_idx = 1
    entities_ids = defaultdict(lambda: "T" + str(len(entities_ids) + 1))
    if not fs.exists(ann_filename) or overwrite_ann:
        with fs.open(ann_filename, "w", encoding="utf-8") as f:
            if "entities" in doc:
                for entity in doc["entities"]:
                    spans = []
                    brat_entity_id = entities_ids[entity["entity_id"]]
                    for fragment in sorted(
                        entity["fragments"], key=lambda frag: frag["begin"]
                    ):
                        idx = fragment["begin"]
                        entity_text = doc["text"][fragment["begin"] : fragment["end"]]
                        # eg: "mon entité  \n  est problématique"
                        for match in re.finditer(
                            r"\s*(.+?)(?:( *\n+)+ *|$)", entity_text, flags=re.DOTALL
                        ):
                            spans.append((idx + match.start(1), idx + match.end(1)))
                    print(
                        "{}\t{} {}\t{}".format(
                            brat_entity_id,
                            str(entity["label"]),
                            ";".join(" ".join(map(str, span)) for span in spans),
                            " ".join(doc["text"][begin:end] for begin, end in spans),
                        ),
                        file=f,
                    )
                    if "attributes" in entity:
                        for i, (key, value) in enumerate(entity["attributes"].items()):
                            if value:
                                print(
                                    "A{}\t{} {}{}".format(
                                        attribute_idx,
                                        str(key),
                                        brat_entity_id,
                                        (" " + str(value)) if value is not True else "",
                                    ),
                                    file=f,
                                )
                                attribute_idx += 1

                    # fmt: off
                    # if "relations" in doc:
                    #     for i, relation in enumerate(doc["relations"]):
                    #         entity_from = entities_ids[relation["from_entity_id"]]
                    #         entity_to = entities_ids[relation["to_entity_id"]]
                    #         print(
                    #             "R{}\t{} Arg1:{} Arg2:{}\t".format(
                    #                 i + 1, str(relation["label"]), entity_from,
                    #                 entity_to
                    #             ),
                    #             file=f,
                    #         )
                    # fmt: on


class StandoffReader(FileBasedReader):
    DATA_FIELDS = ()

    def __init__(
        self,
        path: Union[str, Path],
        *,
        keep_ipynb_checkpoints: bool = False,
        keep_txt_only_docs: bool = False,
        filesystem: Optional[FileSystem] = None,
        loop: bool = False,
        shuffle: Literal["dataset", False] = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.shuffle = shuffle
        self.emitted_sentinels = {"dataset"}
        seed = seed if seed is not None else random.getrandbits(32)
        self.rng = random.Random(seed)
        self.loop = loop
        self.fs, self.path = normalize_fs_path(filesystem, path)
        files = {
            file
            for file in walk_match(self.fs, self.path, ".*[.](txt|a*)")
            if (keep_ipynb_checkpoints or ".ipynb_checkpoints" not in str(file))
        }
        ann_files = {}
        for f in files:
            name, ext = os.path.splitext(f)
            if ext.startswith(".a"):
                ann_files.setdefault(name, []).append(f)
        self.files = sorted(
            [
                (file, ann_files.get(file.replace(".txt", ""), []))
                for file in files
                if file.endswith(".txt")
                and (keep_txt_only_docs or file.replace(".txt", "") in ann_files)
            ]
        )
        assert len(self.files), f"No .txt files found in the BRAT directory {self.path}"
        logger.info(f"The BRAT directory contains {len(self.files)} .txt files.")

    def read_records(self) -> Iterable[Any]:
        while True:
            files = self.files
            if self.shuffle:
                files = shuffle(files, self.rng)
            for item in files:
                txt_path, ann_paths = item
                anns = parse_standoff_file(txt_path, ann_paths, fs=self.fs)
                anns[FILENAME] = os.path.relpath(txt_path, self.path).rsplit(".", 1)[0]
                anns["doc_id"] = anns[FILENAME]
                yield anns
            yield DatasetEndSentinel()
            if not self.loop:
                break

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"path={self.path!r}, "
            f"shuffle={self.shuffle}, "
            f"loop={self.loop})"
        )


class StandoffWriter(BaseWriter):
    def __init__(
        self,
        path: Union[str, Path],
        *,
        lines: bool = True,
        overwrite: bool = False,
        filesystem: Optional[FileSystem] = None,
    ):
        self.fs, self.path = normalize_fs_path(filesystem, path)
        self.lines = lines

        if self.fs.exists(self.path):
            unsafe_exts = Counter(
                os.path.splitext(f)[1]
                for f in walk_match(self.fs, self.path, ".*[.](txt|a.*)")
            )
            if unsafe_exts and not overwrite:
                raise FileExistsError(
                    f"Directory {self.path} already exists and appear to contain "
                    "annotations:"
                    + "".join(f"\n -{s[1:]}: {v} files" for s, v in unsafe_exts.items())
                    + "\nUse overwrite=True to write files anyway."
                )
        self.fs.makedirs(self.path, exist_ok=True)

        super().__init__()

    def handle_record(self, record: Union[Dict, List[Dict]]):
        # TODO support write_in_worker = False (default is True ATM)
        results = []
        for rec in flatten(record):
            filename = str(rec[FILENAME])
            path = os.path.join(self.path, f"{filename}.txt")
            dump_standoff_file(
                rec,
                path,
                overwrite_txt=True,
                overwrite_ann=True,
                fs=self.fs,
            )
            results.append(path)
        return results

    def consolidate(self, items: Iterable[Any]):
        return list(flatten(items))


# noinspection PyIncorrectDocstring
@registry.readers.register("standoff")
def read_standoff(
    path: Union[str, Path],
    *,
    keep_ipynb_checkpoints: bool = False,
    keep_txt_only_docs: bool = False,
    converter: Optional[AsList[Union[str, Callable]]] = ["standoff"],
    filesystem: Optional[FileSystem] = None,
    shuffle: Literal["dataset", False] = False,
    seed: Optional[int] = None,
    loop: bool = False,
    **kwargs,
) -> Stream:
    """
    The BratReader (or `edsnlp.data.read_standoff`) reads a directory of BRAT files and
    yields documents. At the moment, only entities and attributes are loaded. Relations
     and events are not supported.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)
    doc_iterator = edsnlp.data.read_standoff("path/to/brat/directory")
    annotated_docs = nlp.pipe(doc_iterator)
    ```

    !!! note "Generator vs list"

        `edsnlp.data.read_standoff` returns a
        [Stream][edsnlp.core.stream.Stream].
        To iterate over the documents multiple times efficiently or to access them by
        index, you must convert it to a list :

        ```{ .python .no-check }
        docs = list(edsnlp.data.read_standoff("path/to/brat/directory"))
        ```

    !!! warning "True/False attributes"

        Boolean values are not supported by the BRAT editor, and are stored as empty
        (key: empty value) if true, and not stored otherwise. This means that False
        values will not be assigned to attributes by default, which can be problematic
        when deciding if an entity is negated or not : is the entity not negated, or
        has the negation attribute not been annotated ?

        To avoid this issue, you can use the `bool_attributes` argument to specify
        which attributes should be considered as boolean when reading a BRAT dataset.
        These attributes will be assigned a value of `True` if they are present, and
        `False` otherwise.

        ```{ .python .no-check }
        doc_iterator = edsnlp.data.read_standoff(
            "path/to/brat/directory",
            span_attributes=["negation", "family"],
            bool_attributes=["negation"],  # Missing values will be set to False
        )
        ```

    Parameters
    ----------
    path : Union[str, Path]
        Path to the directory containing the BRAT files (will recursively look for
        files in subdirectories).
    shuffle: Literal["dataset", False]
        Whether to shuffle the data. If "dataset", the whole dataset will be shuffled
        before starting iterating on it (at the start of every epoch if looping).
    seed: Optional[int]
        The seed to use for shuffling.
    loop: bool
        Whether to loop over the data indefinitely.
    nlp : Optional[PipelineProtocol]
        The pipeline object (optional and likely not needed, prefer to use the
        `tokenizer` directly argument instead).
    tokenizer : Optional[spacy.tokenizer.Tokenizer]
        The tokenizer instance used to tokenize the documents. Likely not needed since
        by default it uses the current context tokenizer :

        - the tokenizer of the next pipeline run by `.map_pipeline` in a
          [Stream][edsnlp.core.stream.Stream].
        - or the `eds` tokenizer by default.
    span_setter : SpanSetterArg
        The span setter to use when setting the spans in the documents. Defaults to
        setting the spans in the `ents` attribute, and creates a new span group for
        each JSON entity label.
    span_attributes : Optional[AttributesMappingArg]
        Mapping from BRAT attributes to Span extensions (can be a list too).
        By default, all attributes are imported as Span extensions with the same name.
    keep_raw_attribute_values : bool
        Whether to keep the raw attribute values (as strings) or to convert them to
        Python objects (e.g. booleans).
    default_attributes : AttributesMappingArg
        How to set attributes on spans for which no attribute value was found in the
        input format. This is especially useful for negation, or frequent attributes
        values (e.g. "negated" is often False, "temporal" is often "present"), that
        annotators may not want to annotate every time.
    notes_as_span_attribute : Optional[str]
        If set, the AnnotatorNote annotations will be concatenated and stored in a span
        attribute with this name.
    split_fragments : bool
        Whether to split the fragments into separate spans or not. If set to False, the
        fragments will be concatenated into a single span.
    keep_ipynb_checkpoints : bool
        Whether to keep the files that are in the `.ipynb_checkpoints` directory.
    keep_txt_only_docs : bool
        Whether to keep the `.txt` files that do not have corresponding `.ann` files.
    converter : Optional[AsList[Union[str, Callable]]]
        Converter to use to convert the documents to dictionary objects.
    filesystem: Optional[FileSystem] = None,
        The filesystem to use to write the files. If None, the filesystem will be
        inferred from the path (e.g. `s3://` will use S3).

    Returns
    -------
    Stream
    """
    data = Stream(
        reader=StandoffReader(
            path,
            keep_ipynb_checkpoints=keep_ipynb_checkpoints,
            keep_txt_only_docs=keep_txt_only_docs,
            filesystem=filesystem,
            loop=loop,
            shuffle=shuffle,
            seed=seed,
        )
    )
    if converter:
        for conv in converter:
            conv, kwargs = get_dict2doc_converter(conv, kwargs)
            data = data.map(conv, kwargs=kwargs)
    return data


@registry.writers.register("standoff")
def write_standoff(
    data: Union[Any, Stream],
    path: Union[str, Path],
    overwrite: bool = False,
    filesystem: Optional[FileSystem] = None,
    execute: bool = True,
    converter: Optional[Union[str, Callable]] = "standoff",
    **kwargs,
) -> None:
    """
    `edsnlp.data.write_standoff` writes a list of documents using the BRAT/Standoff
    format in a directory. The BRAT files will be named after the `note_id` attribute of
    the documents, and subdirectories will be created if the name contains `/`
    characters.

    Example
    -------
    ```{ .python .no-check }

    import edsnlp

    nlp = edsnlp.blank("eds")
    nlp.add_pipe(...)

    doc = nlp("My document with entities")

    edsnlp.data.write_standoff([doc], "path/to/brat/directory")
    ```

    !!! warning "Overwriting files"

        By default, `write_standoff` will raise an error if the directory already exists
        and contains files with `.a*` or `.txt` suffixes. This is to avoid overwriting
        existing annotations. To allow overwriting existing files, use `overwrite=True`.

    Parameters
    ----------
    data: Union[Any, Stream],
        The data to write (either a list of documents or a Stream).
    path: Union[str, Path]
        Path to the directory containing the BRAT files (will recursively look for
        files in subdirectories).
    span_getter: SpanGetterArg
        The span getter to use when listing the spans that will be exported as BRAT
        entities. Defaults to getting the spans in the `ents` attribute.
    span_attributes: Optional[AttributesMappingArg]
        Mapping from BRAT attributes to Span extension. By default, no attribute will
        be exported.
    overwrite: bool
        Whether to overwrite existing directories.
    filesystem: Optional[FileSystem] = None,
        The filesystem to use to write the files. If None, the filesystem will be
        inferred from the path (e.g. `s3://` will use S3).
    execute: bool
        Whether to execute the writing operation immediately or to return a stream
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the documents to dictionary objects.
        Defaults to the "standoff" format converter.
    """
    data = Stream.ensure_stream(data)
    if converter:
        converter, kwargs = get_doc2dict_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)

    return data.write(
        StandoffWriter(
            path,
            overwrite=overwrite,
            filesystem=filesystem,
        ),
        execute=execute,
    )
