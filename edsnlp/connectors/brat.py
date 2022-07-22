import glob
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from spacy import Language
from spacy.tokens import Doc, Span
from spacy.util import filter_spans
from tqdm import tqdm

REGEX_ENTITY = re.compile(r"^(T\d+)\t([^\s]+)([^\t]+)\t(.*)$")
REGEX_NOTE = re.compile(r"^(#\d+)\tAnnotatorNotes ([^\t]+)\t(.*)$")
REGEX_RELATION = re.compile(r"^(R\d+)\t([^\s]+) Arg1:([^\s]+) Arg2:([^\s]+)")
REGEX_ATTRIBUTE = re.compile(r"^([AM]\d+)\t(.+)$")
REGEX_EVENT = re.compile(r"^(E\d+)\t(.+)$")
REGEX_EVENT_PART = re.compile(r"([^\s]+):([TE]\d+)")


def load_from_brat(path: str, merge_spaced_fragments: bool = True) -> Dict:
    """
    Load a brat file

    Adapted from
    https://github.com/percevalw/nlstruct/blob/master/nlstruct/datasets/brat.py

    Parameters
    ----------
    path: str
        Path or glob path of the brat text file (.txt, not .ann)
    merge_spaced_fragments: bool
        Merge fragments of a entity that was splitted by brat because it overlapped an
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

    # doc_id = filename.replace('.txt', '').split("/")[-1]

    with open(path) as f:
        text = f.read()

    note_id = path.split("/")[-1].rsplit(".", 1)[0]

    if not len(ann_filenames):
        return {
            "note_id": note_id,
            "text": text,
        }

    for ann_file in ann_filenames:
        with open(ann_file) as f:
            for line_idx, line in enumerate(f):
                try:
                    if line.startswith("T"):
                        match = REGEX_ENTITY.match(line)
                        if match is None:
                            raise ValueError(
                                f"File {ann_file}, unrecognized Brat line {line}"
                            )
                        ann_id = match.group(1)
                        entity = match.group(2)
                        span = match.group(3)
                        mention_text = match.group(4)
                        entities[ann_id] = {
                            "text": mention_text,
                            "entity_id": ann_id,
                            "fragments": [],
                            "attributes": [],
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
                            raise ValueError(
                                f"File {ann_file}, unrecognized Brat line {line}"
                            )
                        ann_id = match.group(1)
                        parts = match.group(2).split(" ")
                        if len(parts) >= 3:
                            entity, entity_id, value = parts
                        elif len(parts) == 2:
                            entity, entity_id = parts
                            value = None
                        else:
                            raise ValueError(
                                f"File {ann_file}, unrecognized Brat line {line}"
                            )
                        (
                            entities[entity_id]
                            if entity_id.startswith("T")
                            else events[entity_id]
                        )["attributes"].append(
                            {
                                "attribute_id": ann_id,
                                "label": entity,
                                "value": value,
                            }
                        )
                    elif line.startswith("R"):
                        match = REGEX_RELATION.match(line)
                        if match is None:
                            raise ValueError(
                                f"File {ann_file}, unrecognized Brat line {line}"
                            )
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
                            raise ValueError(
                                f"File {ann_file}, unrecognized Brat line {line}"
                            )
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
                            "attributes": [],
                            "arguments": arguments,
                        }
                    elif line.startswith("#"):
                        match = REGEX_NOTE.match(line)
                        if match is None:
                            raise ValueError(
                                f"File {ann_file}, unrecognized Brat line {line}"
                            )
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
        "note_id": note_id,
        "text": text,
        "entities": list(entities.values()),
        "relations": relations,
        "events": list(events.values()),
    }


def export_to_brat(doc, txt_filename, overwrite_txt=False, overwrite_ann=False):
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
                    idx = None
                    spans = []
                    brat_entity_id = entities_ids[entity["entity_id"]]
                    for fragment in sorted(
                        entity["fragments"], key=lambda frag: frag["begin"]
                    ):
                        idx = fragment["begin"]
                        entity_text = doc["text"][fragment["begin"] : fragment["end"]]
                        for part in entity_text.split("\n"):
                            begin = idx
                            end = idx + len(part)
                            idx = end + 1
                            if begin != end:
                                spans.append((begin, end))
                    print(
                        "{}\t{} {}\t{}".format(
                            brat_entity_id,
                            str(entity["label"]),
                            ";".join(" ".join(map(str, span)) for span in spans),
                            entity_text.replace("\n", " "),
                        ),
                        file=f,
                    )
                    if "attributes" in entity:
                        for i, attribute in enumerate(entity["attributes"]):
                            if "value" in attribute and attribute["value"] is not None:
                                print(
                                    "A{}\t{} {} {}".format(
                                        attribute_idx,
                                        str(attribute["label"]),
                                        brat_entity_id,
                                        attribute["value"],
                                    ),
                                    file=f,
                                )
                            else:
                                print(
                                    "A{}\t{} {}".format(
                                        attribute_idx,
                                        str(attribute["label"]),
                                        brat_entity_id,
                                    ),
                                    file=f,
                                )
                            attribute_idx += 1
            if "relations" in doc:
                for i, relation in enumerate(doc["relations"]):
                    entity_from = entities_ids[relation["from_entity_id"]]
                    entity_to = entities_ids[relation["to_entity_id"]]
                    print(
                        "R{}\t{} Arg1:{} Arg2:{}\t".format(
                            i + 1, str(relation["label"]), entity_from, entity_to
                        ),
                        file=f,
                    )


class BratConnector(object):
    """
    Two-way connector with BRAT. Supports entities only.

    Parameters
    ----------
    directory : str
        Directory containing the BRAT files.
    n_jobs : int, optional
        Number of jobs for multiprocessing, by default 1
    """

    def __init__(self, directory: str, n_jobs: int = 1, run_pipe=False):
        self.directory = directory
        self.n_jobs = n_jobs
        self.run_pipe = run_pipe

    def full_path(self, filename: str) -> str:
        return os.path.join(self.directory, filename)

    def load_brat(self) -> List[Dict]:
        """
        Transforms a BRAT folder to a list of spaCy documents.

        Parameters
        ----------
        nlp:
            A spaCy pipeline.

        Returns
        -------
        docs:
            List of spaCy documents, with annotations in the `ents` attribute.
        """

        files = os.listdir(self.directory)
        filenames = [f for f in files if f.endswith(".txt")]

        assert filenames, f"BRAT directory {self.directory} is empty!"
        logger.info(
            f"The BRAT directory contains {len(filenames)} annotated documents."
        )

        iterator = tqdm(filenames, ascii=True, ncols=100, desc="Annotation extraction")
        with iterator:
            annotations = Parallel(n_jobs=self.n_jobs)(
                delayed(load_from_brat)(self.full_path(filename))
                for filename in filenames
            )

        return annotations

    def brat2docs(self, nlp: Language) -> List[Doc]:
        """
        Transforms a BRAT folder to a list of spaCy documents.

        Parameters
        ----------
        nlp:
            A spaCy pipeline.

        Returns
        -------
        docs:
            List of spaCy documents, with annotations in the `ents` attribute.
        """

        annotations = self.load_brat()

        texts = [doc["text"] for doc in annotations]

        docs = []

        if self.run_pipe:
            gold_docs = nlp.pipe(texts, batch_size=50, n_process=self.n_jobs)
        else:
            gold_docs = (nlp.make_doc(t) for t in texts)

        for doc, doc_annotations in tqdm(
            zip(gold_docs, annotations),
            ascii=True,
            ncols=100,
            desc="spaCy conversion",
            total=len(texts),
        ):

            doc._.note_id = doc_annotations["note_id"]

            spans = []
            span_groups = defaultdict(lambda: [])

            for ent in doc_annotations["entities"]:
                for a in ent["attributes"]:
                    if not Span.has_extension(a["label"]):
                        Span.set_extension(a["label"], default=None)
                for fragment in ent["fragments"]:
                    span = doc.char_span(
                        fragment["begin"],
                        fragment["end"],
                        label=ent["label"],
                        alignment_mode="expand",
                    )
                    for a in ent["attributes"]:
                        span._.set(a["label"], a["value"] if a is not None else True)
                    spans.append(span)
                    span_groups[ent["label"]].append(span)

            doc.ents = filter_spans(spans)
            for group_name, group in span_groups.items():
                doc.spans[group_name] = group

            docs.append(doc)

        return docs

    def doc2brat(self, doc: Doc) -> None:
        """
        Writes a spaCy document to file in the BRAT directory.

        Parameters
        ----------
        doc:
            spaCy Doc object. The spans in `ents` will populate the `note_id.ann` file.
        """
        filename = str(doc._.note_id)

        annotations = {
            "entities": [
                {
                    "entity_id": i,
                    "fragments": [
                        {
                            "begin": ent.start_char,
                            "end": ent.end_char,
                        }
                    ],
                    "label": ent.label_,
                }
                for i, ent in enumerate(doc.ents)
            ],
            "text": doc.text,
        }
        export_to_brat(
            annotations,
            self.full_path(f"{filename}.txt"),
            overwrite_txt=False,
            overwrite_ann=True,
        )

    def docs2brat(self, docs: List[Doc]) -> None:
        """
        Writes a list of spaCy documents to file.

        Parameters
        ----------
        docs:
            List of spaCy documents.
        """
        for doc in docs:
            self.doc2brat(doc)

    def get_brat(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reads texts and annotations, and returns two DataFrame objects.
        For backward compatibility

        Returns
        -------
        texts:
            A DataFrame containing two fields, `note_id` and `note_text`
        annotations:
            A DataFrame containing the annotations.
        """

        brat = self.load_brat()

        texts = pd.DataFrame(
            [
                {
                    "note_id": doc["note_id"],
                    "note_text": doc["text"],
                }
                for doc in brat
            ]
        )

        annotations = pd.DataFrame(
            [
                {
                    "note_id": doc["note_id"],
                    "index": i,
                    "begin": f["begin"],
                    "end": f["end"],
                    "label": e["label"],
                    "lexical_variant": e["text"],
                }
                for doc in brat
                for i, e in enumerate(doc["entities"])
                for f in e["fragments"]
            ]
        )

        return texts, annotations
