import copy
import re
from difflib import SequenceMatcher
from typing import List, Tuple

import numpy as np
import regex
from typing_extensions import NotRequired, TypedDict


class DeltaCollection(object):
    """
    Copied from
    https://github.com/percevalw/nlstruct/blob/master/nlstruct/data_utils.py#L13
    """

    def __init__(self, begins, ends, deltas):
        self.begins = np.asarray(begins, dtype=int)
        self.ends = np.asarray(ends, dtype=int)
        self.deltas = np.asarray(deltas, dtype=int)

    def apply(self, positions, side="left"):
        positions = np.asarray(positions, dtype=int)
        to_add = (
            (positions.reshape(-1, 1) >= self.ends.reshape(1, -1)) * self.deltas
        ).sum(axis=1)
        between = np.logical_and(
            self.begins.reshape(1, -1) < positions.reshape(-1, 1),
            positions.reshape(-1, 1) < self.ends.reshape(1, -1),
        )
        between_mask = between.any(axis=1)
        between = between[between_mask]
        if len(between) > 0:
            between_i = between.argmax(axis=1)
            if side == "right":
                to_add[between_mask] += (
                    self.ends[between_i]
                    - positions[between_mask]
                    + self.deltas[between_i]
                )
            elif side == "left":
                to_add[between_mask] += self.begins[between_i] - positions[between_mask]
        return positions + to_add

    def unapply(self, positions, side="left"):
        positions = np.asarray(positions, dtype=int)
        begins = self.apply(self.begins, side="left")
        ends = self.apply(self.ends, side="right")
        to_remove = -(
            (positions.reshape(-1, 1) >= ends.reshape(1, -1)) * self.deltas
        ).sum(axis=1)
        between = np.logical_and(
            begins.reshape(1, -1) < positions.reshape(-1, 1),
            positions.reshape(-1, 1) < ends.reshape(1, -1),
        )
        between_mask = between.any(axis=1)
        between = between[between_mask]
        pos = positions + to_remove
        if len(between) > 0:
            between_i = between.argmax(axis=1)
            if side == "right":
                pos[between_mask] = self.ends[between_i]
            elif side == "left":
                pos[between_mask] = self.begins[between_i]
        return pos

    def __add__(self, other):
        if len(self.begins) == 0:
            return other
        if len(other.begins) == 0:
            return self
        begins = self.unapply(other.begins, side="left")
        ends = self.unapply(other.ends, side="right")
        new_begins = np.concatenate([begins, self.begins])
        new_ends = np.concatenate([ends, self.ends])
        new_deltas = np.concatenate([other.deltas, self.deltas])
        sorter = np.lexsort((new_ends, new_begins))
        return DeltaCollection(new_begins[sorter], new_ends[sorter], new_deltas[sorter])


def make_str_from_groups(replacement, groups):
    for i, group in enumerate(groups):
        group = group or ""
        replacement = replacement.replace(f"\\{i + 1}", group).replace(
            f"\\g<{i + 1}>", group
        )
    return replacement


def regex_sub_with_spans(pattern, replacement, text):
    needed_groups = [
        int(next(j for j in i if j))
        for i in regex.findall(r"\\([0-9]+)|\\g<([0-9]+)>", replacement)
    ]
    begins = []
    ends = []
    deltas = []
    for match in reversed(list(regex.finditer(pattern, text, flags=regex.DOTALL))):
        middle = make_str_from_groups(
            replacement, [match.group(i) for i in needed_groups]
        )
        start = match.start()
        end = match.end()
        text = text[:start] + middle + text[end:]
        begins.append(start)
        ends.append(end)
        deltas.append(len(middle) - end + start)
    return text, DeltaCollection(begins, ends, deltas)


def regex_multisub_with_spans(
    patterns, replacements, text, deltas=None, return_deltas=False
):
    deltas = DeltaCollection([], [], []) if deltas is None and return_deltas else None
    for pattern, replacement in zip(patterns, replacements):
        if return_deltas:
            text, new_deltas = regex_sub_with_spans(pattern, replacement, text)
            deltas += new_deltas
        else:
            text = regex.sub(pattern, replacement, text)
    return text, deltas


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_left_context(fragment, text, n=20):
    return re.sub(r"\s+", " ", text[: fragment["begin"]])[-n:]


def get_right_context(fragment, text, n=20):
    return re.sub(r"\s+", " ", text[fragment["end"] :])[:n]


def remove_spaces(doc, shift=False):
    doc = copy.deepcopy(doc)
    doc["text"], deltas = regex_multisub_with_spans(
        [r"\s+"], [" "], doc["text"], return_deltas=True
    )
    if shift:
        doc = shift_spans(doc, deltas)
    return doc, deltas


def shift_spans(doc, deltas, reverse=False):
    fragments = [f for e in doc["entities"] for f in e["fragments"]]
    lefts = [f["begin"] for f in fragments]
    rights = [f["end"] for f in fragments]
    if not reverse:
        new_lefts = deltas.apply(lefts, side="left")
        new_rights = deltas.apply(rights, side="right")
    else:
        new_lefts = deltas.unapply(lefts, side="left")
        new_rights = deltas.unapply(rights, side="right")
    for f, b, e in zip(fragments, new_lefts.tolist(), new_rights.tolist()):
        f["begin"] = b
        f["end"] = e
    return doc


Fragment = TypedDict("Fragment", {"begin": int, "end": int})

AnnotatedText = TypedDict(
    "AnnotatedText",
    {
        "doc_id": NotRequired[str],
        "text": str,
        "entities": List[
            TypedDict(
                "Entity",
                {"label": str, "fragments": List[Fragment]},
            )
        ],
    },
)

Result = TypedDict(
    "Result",
    {
        "total_count": int,
        "missing": List[Fragment],
        "good": List[Fragment],
        "unsure": List[Fragment],
        "missing_count": int,
        "good_count": int,
        "unsure_count": int,
        "doc": AnnotatedText,
    },
)


def align(
    old: AnnotatedText,
    new: AnnotatedText,
    sim_scheme: List[Tuple[int, float]] = [
        (20, 0.70),
        (50, 0.20),
        (100, 0.15),
        (400, 0.10),
        (1000, 0.05),
    ],
    threshold: float = 1.0,
    do_debug: bool = False,
) -> Result:
    """
    Align entities from two similar, but not identical documents. The entities
    of the old document are aligned to the new document based on string matching
    and context similarity, and annotated to the new document.

    Unlike approaches relying on diffing algorithms, this method can handle
    insertions and deletions and swaps of text blocks.

    This method was developed during our work on
    [edspdf](https://github.com/aphp/edspdf), to transfer annotations
    from dataset annotated on previous versions of documents to
    newer versions of the same documents. Yet, there is significant
    room for improvement, especially in the similarity scoring scheme.


    Parameters
    ----------
    old : AnnotatedText
        The old document with entities to align.
    new : AnnotatedText
        The new document to align entities to.
    sim_scheme : List[Tuple[int, float]]
        A list of (context size, weight) tuples to use for similarity scoring.
        Each tuple defines how many characters of left and right context to consider
        and the weight of that context in the overall similarity score.
    threshold : float
        The similarity score threshold above which a match is considered good.
        It's computed as a weighted sum of left and right similarity scores from
        different context sizes.
    do_debug : bool
        Whether to print debug information.

    Returns
    -------
    Result
    """
    missing = []
    good = []
    unsure = []

    # Normalize spaces from both docs (and keep track of character shifts)
    old, _ = remove_spaces(old, shift=True)
    new_raw_text = new["text"]
    new, deltas = remove_spaces(new)

    if do_debug:
        debug = print
        print("Doc ID:", new.get("doc_id", None))
    else:

        def debug(*args, **kwargs):  # pragma: no cover
            return None

    old_text = old["text"]
    new_text = new["text"]

    # List fragments in old doc
    old_fragments = [
        {"label": e["label"], **f} for e in old["entities"] for f in e["fragments"]
    ]
    new_fragments = []

    # Try and find offsets in the new document for every old fragment
    for fragment in old_fragments:
        begin_is_end = False
        if (fragment["begin"], fragment["end"]) == (0, 0):  # pragma: no cover
            new_fragments.append(fragment)
            continue

        fragment_text = old_text[fragment["begin"] : fragment["end"]]
        original_fragment_text = fragment_text

        if len(fragment_text) <= 2:
            try:
                fragment_text = old_text[fragment["begin"] :].split()[0]
            except IndexError:  # pragma: no cover
                fragment_text = ""

        if len(fragment_text) <= 2:
            fragment_text = old_text[fragment["begin"] : fragment["begin"] + 10]

        virtual_fragment = {
            **fragment,
            "end": fragment["begin"] + len(fragment_text),
        }
        snippets = [
            (
                get_left_context(virtual_fragment, old_text, n),
                get_right_context(virtual_fragment, old_text, n),
            )
            for n, w in sim_scheme
        ]

        candidates = []
        for match in re.finditer(re.escape(fragment_text), new["text"]):
            candidates.append({"begin": match.start(), "end": match.end()})

        if len(original_fragment_text) <= 2 and len(candidates) > 20:
            debug(
                "Too small: {}[{}]{}".format(
                    snippets[0][0], fragment_text, snippets[0][1]
                )
            )
            missing.append(fragment)
            new_fragments.append(None)
            continue

        if len(candidates) == 0:
            begin_is_end = True
            prev_snippets = snippets
            prev_fragment_text = fragment_text
            fragment_text = old_text[fragment["end"] : fragment["end"] + 10]
            virtual_fragment = {
                **fragment,
                "begin": fragment["end"],
                "end": fragment["end"] + 10,
            }
            snippets = [
                (
                    get_left_context(virtual_fragment, old_text, n),
                    get_right_context(virtual_fragment, old_text, n),
                )
                for n, w in sim_scheme
            ]
            candidates = []
            for match in re.finditer(re.escape(fragment_text), new["text"]):
                candidates.append({"begin": match.start(), "end": match.end()})

            if len(original_fragment_text) <= 2 and len(candidates) > 20:
                debug(
                    "Too small: {}[{}]{}".format(
                        snippets[0][0], fragment_text, snippets[0][1]
                    ),
                    prev_fragment_text,
                )
                missing.append(fragment)
                new_fragments.append(None)
                continue

            if len(candidates) == 0:
                debug(
                    "! Missing !: {}[{}]{}".format(
                        prev_snippets[0][0], prev_fragment_text, prev_snippets[0][1]
                    )
                )
                missing.append(fragment)
                new_fragments.append(None)
                continue

        for candidate in candidates:
            new_snippets = [
                (
                    get_left_context(candidate, new_text, n),
                    get_right_context(candidate, new_text, n),
                )
                for n, w in sim_scheme
            ]
            score = sum(
                (similar(old_l, new_l) + similar(old_r, new_r)) * w
                for ((old_l, old_r), (new_l, new_r), (n, w)) in zip(
                    snippets, new_snippets, sim_scheme
                )
            ) / sum(w for n, w in sim_scheme)

            candidate["score"] = score
            candidate["new_snippets"] = new_snippets

        best_score = max((c["score"] for c in candidates))

        if begin_is_end:
            snippets = prev_snippets
            fragment_text = prev_fragment_text

        if best_score > threshold:
            good.append(fragment)
            candidate = max(candidates, key=lambda c: c["score"])
            if begin_is_end:
                new_fragments.append(
                    {
                        **fragment,
                        "begin": candidate["begin"]
                        - (fragment["end"] - fragment["begin"]),
                        "end": candidate["begin"],
                    }
                )
            else:
                new_fragments.append(
                    {
                        **fragment,
                        "begin": candidate["begin"],
                        "end": candidate["begin"] + fragment["end"] - fragment["begin"],
                    }
                )
        else:
            unsure.append(fragment)
            new_fragments.append(None)

            debug("Unsure about ", repr(fragment_text))
            debug(
                "    {}[{}]{}".format(snippets[0][0], fragment_text, snippets[0][1])
                .ljust(50)
                .replace("\n", " "),
                "(ORIGINAL)",
            )
            debug("    ------ ↓ candidates ↓ ------")
            for candidate in candidates:
                new_snippets = candidate["new_snippets"]
                score_str = (
                    str(candidate["score"])
                    if candidate["score"] != best_score
                    else "\033[91;1m{}\n\033[0m".format(str(candidate["score"]))
                )
                debug(
                    "    {}[{}]{} : {}".format(
                        new_snippets[0][0],
                        fragment_text,
                        new_snippets[0][1],
                        score_str,
                    ).replace("\n", " ")
                )
            debug()

    # Reconstruct new document from new fragments
    new_fragments_iter = iter(new_fragments)
    new_doc = {
        **old,
        "text": new_raw_text,
        "entities": [
            {
                **e,
                "fragments": [
                    {"begin": new_f["begin"], "end": new_f["end"]}
                    for old_f, new_f in zip(e["fragments"], new_fragments_iter)
                    if new_f is not None
                ],
            }
            for e in old["entities"]
        ],
    }
    new_doc["entities"] = [
        e for e in new_doc["entities"] if all(f is not None for f in e["fragments"])
    ]
    # shift it since we have offsets in the new document with normalized spaces
    new_doc = shift_spans(new_doc, deltas, reverse=True)

    return {
        "total_count": len(old_fragments),
        "missing": missing,
        "good": good,
        "unsure": unsure,
        "missing_count": len(missing),
        "good_count": len(good),
        "unsure_count": len(unsure),
        "doc": new_doc,
    }
