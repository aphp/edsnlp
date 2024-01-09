# ruff: noqa: F401
import glob
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Union,
)

import spacy.tokenizer
from loguru import logger

from edsnlp import registry
from edsnlp.core import PipelineProtocol
from edsnlp.core.lazy_collection import LazyCollection
from edsnlp.data.base import BaseReader, BaseWriter
from edsnlp.data.converters import (
    FILENAME,
    AttributesMappingArg,
    SequenceStr,
    get_dict2doc_converter,
    get_doc2dict_converter,
)
from edsnlp.utils.collections import flatten_once
from edsnlp.utils.span_getters import SpanSetterArg

REGEX_ENTITY = re.compile(r"^(T\d+)\t(.*) (\d+ \d+(?:;\d+ \d+)*)\t(.*)$")
REGEX_NOTE = re.compile(r"^(#\d+)\tAnnotatorNotes ([^\t]+)\t(.*)$")
REGEX_RELATION = re.compile(r"^(R\d+)\t(\S+) Arg1:(\S+) Arg2:(\S+)")
REGEX_ATTRIBUTE = re.compile(r"^([AM]\d+)\t(.+?) ([TE]\d+)(?: (.+))?$")
REGEX_EVENT = re.compile(r"^(E\d+)\t(.+)$")
REGEX_EVENT_PART = re.compile(r"(\S+):([TE]\d+)")


class BratParsingError(ValueError):
    def __init__(self, ann_file, line):
        super().__init__(f"File {ann_file}, unrecognized Brat line {line}")


def parse_standoff_file(path: str, merge_spaced_fragments: bool = True) -> Dict:
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

    Returns
    -------
    Iterator[Dict]
    """
    ann_filenames = []
    for filename in glob.glob(path.replace(".txt", ".a*"), recursive=True):
        ann_filenames.append(filename)

    entities = {}
    relations = []
    events = {}

    with open(path) as f:
        text = f.read()

    if not len(ann_filenames):
        return {
            "text": text,
        }

    for ann_file in ann_filenames:
        with open(ann_file) as f:
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
                            "comments": [],
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
                        comment = match.group(3)
                        entities[entity_id]["comments"].append(
                            {
                                "comment_id": ann_id,
                                "comment": comment,
                            }
                        )
                except Exception:
                    raise Exception(
                        "Could not parse line {} from {}: {}".format(
                            line_idx, filename.replace(".txt", ".ann"), repr(line)
                        )
                    )
    return {
        "text": text,
        "entities": list(entities.values()),
        "relations": relations,
        "events": list(events.values()),
    }


def dump_standoff_file(doc, txt_filename, overwrite_txt=False, overwrite_ann=False):
    parent_dir = txt_filename.rsplit("/", 1)[0]
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    if not os.path.exists(txt_filename) or overwrite_txt:
        with open(txt_filename, "w") as f:
            f.write(doc["text"])

    ann_filename = txt_filename.replace(".txt", ".ann")
    attribute_idx = 1
    entities_ids = defaultdict(lambda: "T" + str(len(entities_ids) + 1))
    if not os.path.exists(ann_filename) or overwrite_ann:
        with open(ann_filename, "w") as f:
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


class StandoffReader(BaseReader):
    DATA_FIELDS = ()

    def __init__(
        self,
        path: Union[str, Path],
        *,
        keep_ipynb_checkpoints: bool = False,
        keep_txt_only_docs: bool = False,
    ):
        super().__init__()
        self.path = Path(path)
        self.files = [
            file
            for file in self.path.rglob("*.txt")
            if (keep_ipynb_checkpoints or ".ipynb_checkpoints" not in str(file))
            and (keep_txt_only_docs or glob.glob(str(path).replace(".txt", ".a*")))
        ]
        assert len(self.files), f"No .txt files found in the BRAT directory {path}"
        for file in self.files:
            if not file.exists():
                raise FileNotFoundError(f"File {file} does not exist")
        logger.info(f"The BRAT directory contains {len(self.files)} .txt files.")

    def read_main(self):
        return ((f, 1) for f in self.files)

    def read_worker(self, fragment):
        tasks = []
        for file in fragment:
            anns = parse_standoff_file(str(file))
            anns[FILENAME] = str(file.relative_to(self.path)).rsplit(".", 1)[0]
            anns["doc_id"] = anns[FILENAME]
            tasks.append(anns)
        return tasks


class StandoffWriter(BaseWriter):
    def __init__(
        self,
        path: Union[str, Path],
        *,
        lines: bool = True,
        overwrite: bool = False,
    ):
        self.path = path
        self.lines = lines

        if path.exists():
            suffixes = Counter(f.suffix for f in path.iterdir())
            unsafe_suffixes = {
                s: v
                for s, v in suffixes.items()
                if s.startswith(".a") or s.startswith(".txt")
            }
            if unsafe_suffixes and not overwrite:
                raise FileExistsError(
                    f"Directory {path} already exists and appear to contain "
                    "annotations:"
                    + "".join(f"\n -{s}: {v} files" for s, v in unsafe_suffixes.items())
                    + "\nUse overwrite=True to write files anyway."
                )
        path.mkdir(parents=True, exist_ok=True)

        super().__init__()

    def write_worker(self, records):
        # If write as jsonl, we will perform the actual writing in the `write` method
        results = []
        for rec in records:
            filename = str(rec[FILENAME])
            path = str(self.path / f"{filename}.txt")
            dump_standoff_file(
                rec,
                path,
                overwrite_txt=True,
                overwrite_ann=True,
            )
            results.append(path)
        return results, len(results)

    def write_main(self, fragments):
        return list(flatten_once(fragments))


# noinspection PyIncorrectDocstring
@registry.readers.register("standoff")
def read_standoff(
    path: Union[str, Path],
    *,
    keep_ipynb_checkpoints: bool = False,
    keep_txt_only_docs: bool = False,
    converter: Optional[Union[str, Callable]] = "standoff",
    **kwargs,
) -> LazyCollection:
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
        [LazyCollection][edsnlp.core.lazy_collection.LazyCollection].
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
            # Mapping from 'BRAT attribute name' to 'Doc attribute name'
            span_attributes={"Negation": "negated"},
            bool_attributes=["negated"],  # Missing values will be set to False
        )
        ```

    Parameters
    ----------
    path : Union[str, Path]
        Path to the directory containing the BRAT files (will recursively look for
        files in subdirectories).
    nlp : Optional[PipelineProtocol]
        The pipeline object (optional and likely not needed, prefer to use the
        `tokenizer` directly argument instead).
    tokenizer : Optional[spacy.tokenizer.Tokenizer]
        The tokenizer instance used to tokenize the documents. Likely not needed since
        by default it uses the current context tokenizer :

        - the tokenizer of the next pipeline run by `.map_pipeline` in a
          [LazyCollection][edsnlp.core.lazy_collection.LazyCollection].
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
    bool_attributes : SequenceStr
        List of attributes for which missing values should be set to False.

    Returns
    -------
    LazyCollection
    """
    data = LazyCollection(
        reader=StandoffReader(
            path,
            keep_ipynb_checkpoints=keep_ipynb_checkpoints,
            keep_txt_only_docs=keep_txt_only_docs,
        )
    )
    if converter:
        converter, kwargs = get_dict2doc_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)
    return data


@registry.writers.register("standoff")
def write_standoff(
    data: Union[Any, LazyCollection],
    path: Union[str, Path],
    overwrite: bool = False,
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
    data: Union[Any, LazyCollection],
        The data to write (either a list of documents or a LazyCollection).
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
    converter: Optional[Union[str, Callable]]
        Converter to use to convert the documents to dictionary objects.
        Defaults to the "standoff" format converter.
    """
    data = LazyCollection.ensure_lazy(data)
    if converter:
        converter, kwargs = get_doc2dict_converter(converter, kwargs)
        data = data.map(converter, kwargs=kwargs)

    return data.write(StandoffWriter(path, overwrite=overwrite))
