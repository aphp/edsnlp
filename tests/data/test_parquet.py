from itertools import islice
from pathlib import Path

import pyarrow.dataset
import pyarrow.fs
import pytest
from confit.utils.random import set_seed
from typing_extensions import Literal

import edsnlp
from edsnlp.data.converters import get_dict2doc_converter, get_doc2dict_converter
from edsnlp.utils.collections import dl_to_ld


def assert_doc_read(doc):
    assert doc._.note_id == "subfolder/doc-1"
    assert doc._.context_var == "test"

    attrs = ("etat", "assertion")
    spans_and_attributes = {
        "__ents__": sorted(
            [
                (e.start, e.end, e.text, tuple(getattr(e._, key) for key in attrs))
                for e in doc.ents
            ]
        ),
        **{
            name: sorted(
                [
                    (e.start, e.end, e.text, tuple(getattr(e._, key) for key in attrs))
                    for e in doc.spans[name]
                ]
            )
            for name in doc.spans
        },
    }

    assert spans_and_attributes == {
        "__ents__": [
            (6, 7, "douleurs", (None, None)),
            (7, 11, "dans le bras droit", (None, None)),
            (17, 21, "problème \nde locomotion", (None, "absent")),
            (25, 26, "AVC", ("passé", "non-associé")),
            (35, 36, "rhume", ("présent", "hypothétique")),
            (45, 46, "rhume", ("présent", "hypothétique")),
            (51, 52, "Douleurs", (None, None)),
            (52, 56, "dans le bras droit", (None, None)),
            (68, 69, "anomalie", (None, "absent")),
        ],
        "anatomie": [
            (9, 11, "bras droit", (None, None)),
            (54, 56, "bras droit", (None, None)),
        ],
        "localisation": [
            (7, 11, "dans le bras droit", (None, None)),
            (52, 56, "dans le bras droit", (None, None)),
        ],
        "pathologie": [
            (17, 21, "problème \nde locomotion", (None, "absent")),
            (25, 26, "AVC", ("passé", "non-associé")),
            (35, 36, "rhume", ("présent", "hypothétique")),
            (45, 46, "rhume", ("présent", "hypothétique")),
        ],
        "sosy": [
            (6, 7, "douleurs", (None, None)),
            (51, 52, "Douleurs", (None, None)),
            (68, 69, "anomalie", (None, "absent")),
        ],
    }


GOLD_OMOP = {
    "entities": [
        {
            "assertion": None,
            "end_char": 38,
            "etat": "test",
            "lexical_variant": "douleurs",
            "note_nlp_id": 0,
            "note_nlp_source_value": "sosy",
            "start_char": 30,
        },
        {
            "assertion": None,
            "end_char": 57,
            "etat": None,
            "lexical_variant": "dans le bras droit",
            "note_nlp_id": 1,
            "note_nlp_source_value": "localisation",
            "start_char": 39,
        },
        {
            "assertion": None,
            "end_char": 57,
            "etat": None,
            "lexical_variant": "bras droit",
            "note_nlp_id": 2,
            "note_nlp_source_value": "anatomie",
            "start_char": 47,
        },
        {
            "assertion": "absent",
            "end_char": 98,
            "etat": None,
            "lexical_variant": "problème \nde locomotion",
            "note_nlp_id": 3,
            "note_nlp_source_value": "pathologie",
            "start_char": 75,
        },
        {
            "assertion": "non-associé",
            "end_char": 117,
            "etat": "passé",
            "lexical_variant": "AVC",
            "note_nlp_id": 4,
            "note_nlp_source_value": "pathologie",
            "start_char": 114,
        },
        {
            "assertion": "hypothétique",
            "end_char": 164,
            "etat": "présent",
            "lexical_variant": "rhume",
            "note_nlp_id": 5,
            "note_nlp_source_value": "pathologie",
            "start_char": 159,
        },
        {
            "assertion": "hypothétique",
            "end_char": 296,
            "etat": "présent",
            "lexical_variant": "rhume",
            "note_nlp_id": 6,
            "note_nlp_source_value": "pathologie",
            "start_char": 291,
        },
        {
            "assertion": None,
            "end_char": 314,
            "etat": None,
            "lexical_variant": "Douleurs",
            "note_nlp_id": 7,
            "note_nlp_source_value": "sosy",
            "start_char": 306,
        },
        {
            "assertion": None,
            "end_char": 333,
            "etat": None,
            "lexical_variant": "dans le bras droit",
            "note_nlp_id": 8,
            "note_nlp_source_value": "localisation",
            "start_char": 315,
        },
        {
            "assertion": None,
            "end_char": 333,
            "etat": None,
            "lexical_variant": "bras droit",
            "note_nlp_id": 9,
            "note_nlp_source_value": "anatomie",
            "start_char": 323,
        },
        {
            "assertion": "absent",
            "end_char": 386,
            "etat": None,
            "lexical_variant": "anomalie",
            "note_nlp_id": 10,
            "note_nlp_source_value": "sosy",
            "start_char": 378,
        },
    ],
    "note_id": "subfolder/doc-1",
    "context_var": "test",
    "note_text": "Le patient est admis pour des douleurs dans le bras droit, mais "
    "n'a pas de problème \n"
    "de locomotion. \n"
    "Historique d'AVC dans la famille. pourrait être un cas de "
    "rhume.\n"
    "NBNbWbWbNbWbNBNbNbWbWbNBNbWbNbNbWbNBNbWbNbNBWbWbNbNbNBWbNbWbNbWBNbNbWbNbNBNbWb"
    "WbNbWBNbNbWbNBNbWbWbNb\n"
    "Pourrait être un cas de rhume.\n"
    "Motif :\n"
    "Douleurs dans le bras droit.\n"
    "ANTÉCÉDENTS\n"
    "Le patient est déjà venu\n"
    "Pas d'anomalie détectée.\n",
}


def assert_doc_write_omop(exported_obj):
    assert exported_obj == GOLD_OMOP


def assert_doc_write_ents(exported_objs):
    in_converter, kwargs = get_dict2doc_converter(
        "omop",
        dict(
            span_attributes=["etat", "assertion"],
            doc_attributes=["context_var"],
        ),
    )
    doc = in_converter(GOLD_OMOP, **kwargs)
    out_converter, kwargs = get_doc2dict_converter(
        "ents",
        dict(
            span_attributes=["etat", "assertion"],
            doc_attributes=["context_var"],
            span_getter=["ents", "sosy", "localisation", "anatomie", "pathologie"],
        ),
    )
    GOLD_ENTS = out_converter(doc, **kwargs)
    assert exported_objs == GOLD_ENTS


def test_read_write_in_worker(blank_nlp, tmpdir):
    input_dir = Path(__file__).parent.parent.resolve() / "resources" / "docs.parquet"
    output_dir = Path(tmpdir)
    edsnlp.data.read_parquet(
        input_dir,
        converter="omop",
        span_attributes=["etat", "assertion"],
        doc_attributes=["context_var"],
        read_in_worker=True,
    ).write_parquet(
        output_dir / "docs.parquet",
        converter="omop",
        doc_attributes=["context_var"],
        span_attributes=["etat", "assertion"],
        span_getter=["ents", "sosy", "localisation", "anatomie", "pathologie"],
        write_in_worker=True,
    )
    # fmt: off
    assert (
            list(dl_to_ld(pyarrow.dataset.dataset(output_dir / "docs.parquet").to_table().to_pydict()))  # noqa: E501
            == list(dl_to_ld(pyarrow.dataset.dataset(input_dir).to_table().to_pydict()))
    )
    # fmt: on


def test_read_to_parquet(blank_nlp, tmpdir, run_in_test_dir):
    output_dir = Path(tmpdir)
    fs = pyarrow.fs.LocalFileSystem()
    doc = list(
        edsnlp.data.read_parquet(
            "../resources/docs.parquet",
            converter="omop",
            span_attributes=["etat", "assertion"],
            doc_attributes=["context_var"],
            filesystem=fs,
        )
    )[0]
    assert_doc_read(doc)
    doc.ents[0]._.etat = "test"

    edsnlp.data.write_parquet(
        [doc],
        output_dir,
        converter="omop",
        doc_attributes=["context_var"],
        span_attributes=["etat", "assertion"],
        span_getter=["ents", "sosy", "localisation", "anatomie", "pathologie"],
    )

    assert_doc_write_omop(
        next(dl_to_ld(pyarrow.dataset.dataset(output_dir).to_table().to_pydict()))
    )

    with pytest.raises(FileExistsError):
        edsnlp.data.write_parquet(
            [doc],
            output_dir,
            converter="omop",
            doc_attributes=["context_var"],
            span_attributes=["etat", "assertion"],
            span_getter=["ents", "sosy", "localisation", "anatomie", "pathologie"],
        )

    edsnlp.data.write_parquet(
        [doc],
        output_dir,
        converter="omop",
        doc_attributes=["context_var"],
        span_attributes=["etat", "assertion"],
        span_getter=["ents", "sosy", "localisation", "anatomie", "pathologie"],
        overwrite=True,
    )


def test_read_to_parquet_ents(blank_nlp, tmpdir):
    input_dir = Path(__file__).parent.parent.resolve() / "resources" / "docs.parquet"
    output_dir = Path(tmpdir)
    fs = pyarrow.fs.LocalFileSystem()
    doc = list(
        edsnlp.data.read_parquet(
            input_dir,
            converter="omop",
            span_attributes=["etat", "assertion"],
            doc_attributes=["context_var"],
            filesystem=fs,
        )
    )[0]
    assert_doc_read(doc)
    doc.ents[0]._.etat = "test"

    edsnlp.data.write_parquet(
        [doc],
        output_dir,
        converter="ents",
        doc_attributes=["context_var"],
        span_attributes=["etat", "assertion"],
        span_getter=["ents", "sosy", "localisation", "anatomie", "pathologie"],
        num_rows_per_file=1024,  # deprecated but test for backward compatibility
    )

    assert_doc_write_ents(
        list(dl_to_ld(pyarrow.dataset.dataset(output_dir).to_table().to_pydict()))
    )

    with pytest.raises(FileExistsError):
        edsnlp.data.write_parquet(
            [doc],
            output_dir,
            converter="ents",
            doc_attributes=["context_var"],
            span_attributes=["etat", "assertion"],
            span_getter=["ents", "sosy", "localisation", "anatomie", "pathologie"],
        )


@pytest.mark.parametrize("num_cpu_workers", [0, 2])
@pytest.mark.parametrize("shuffle", ["dataset", "fragment"])
@pytest.mark.parametrize("shuffle_reader", [False, None])
def test_read_shuffle_loop(
    num_cpu_workers: int,
    shuffle: Literal["dataset", "fragment"],
    shuffle_reader: bool,
):
    input_dir = Path(__file__).parent.parent.resolve() / "resources" / "docs.parquet"
    notes = (
        edsnlp.data.read_parquet(input_dir, loop=True)
        .shuffle(batch_by=shuffle, seed=42, shuffle_reader=shuffle_reader)
        .map(lambda x: x["note_id"])
        .set_processing(num_cpu_workers=num_cpu_workers)
    )
    notes = list(islice(notes, 6))
    if not (num_cpu_workers > 1 and not shuffle_reader):
        assert notes == [
            "subfolder/doc-2",
            "subfolder/doc-1",
            "subfolder/doc-3",
            "subfolder/doc-3",
            "subfolder/doc-2",
            "subfolder/doc-1",
        ]


@pytest.mark.parametrize("num_cpu_workers", [0, 2])
@pytest.mark.parametrize("work_unit", ["record", "fragment"])
@pytest.mark.parametrize("shuffle", [False, "dataset", "fragment"])
def test_read_work_unit(
    num_cpu_workers,
    work_unit: Literal["record", "fragment"],
    shuffle: Literal[False, "dataset", "fragment"],
):
    if shuffle == "dataset" and work_unit == "fragment":
        pytest.skip("Dataset-level shuffle is not supported with fragment work unit")
    input_dir = Path(__file__).parent.parent.resolve() / "resources" / "docs.parquet"
    set_seed(42)
    stream = edsnlp.data.read_parquet(
        input_dir, work_unit=work_unit, shuffle=shuffle
    ).set_processing(
        num_cpu_workers=num_cpu_workers,
    )
    stream = stream.map_batches(
        lambda b: "|".join(sorted([x["note_id"] for x in b])), batch_size=1000
    )
    if work_unit == "fragment" and num_cpu_workers == 2 or num_cpu_workers == 0:
        assert list(stream) == ["subfolder/doc-1|subfolder/doc-2|subfolder/doc-3"]
    else:
        assert list(stream) == ["subfolder/doc-1|subfolder/doc-3", "subfolder/doc-2"]


@pytest.mark.parametrize(
    "num_cpu_workers,write_in_worker",
    [
        (0, False),
        (2, True),
        (2, False),
    ],
)
def test_write_parquet_fragment(tmpdir, num_cpu_workers, write_in_worker):
    input_dir = Path(__file__).parent.parent.resolve() / "resources" / "docs.parquet"
    output_dir = Path(tmpdir)
    notes = edsnlp.data.read_parquet(input_dir, converter="omop")
    notes = notes.map_batches(
        lambda b: [y for x in b for y in (x,) * (4 if "doc-2" in x._.note_id else 1)]
    )
    notes = notes.set_processing(
        num_cpu_workers=num_cpu_workers,
        deterministic=True,  # by default
    )
    notes.write_parquet(
        output_dir,
        batch_size="fragment",
        converter=lambda x: {"note_id": x._.note_id},
        write_in_worker=write_in_worker,
    )
    input_ds = pyarrow.dataset.dataset(input_dir)
    inputs = [o["note_id"] for o in dl_to_ld(input_ds.to_table().to_pydict())]
    assert len(list(input_ds.get_fragments())) == 1
    assert inputs == ["subfolder/doc-1", "subfolder/doc-2", "subfolder/doc-3"]
    out_ds = pyarrow.dataset.dataset(output_dir)
    outs = [o["note_id"] for o in dl_to_ld(out_ds.to_table().to_pydict())]
    if write_in_worker and num_cpu_workers == 2:
        # Depending on the order in which the 2 workers produces the batches
        assert len(list(out_ds.get_fragments())) == 2
        assert outs == [
            "subfolder/doc-1",
            "subfolder/doc-3",
            "subfolder/doc-2",
            "subfolder/doc-2",
            "subfolder/doc-2",
            "subfolder/doc-2",
        ] or outs == [
            "subfolder/doc-2",
            "subfolder/doc-2",
            "subfolder/doc-2",
            "subfolder/doc-2",
            "subfolder/doc-1",
            "subfolder/doc-3",
        ]
    elif not write_in_worker and num_cpu_workers == 2:
        assert len(list(out_ds.get_fragments())) == 1
        assert outs == [
            "subfolder/doc-1",
            "subfolder/doc-2",
            "subfolder/doc-3",
            "subfolder/doc-2",
            "subfolder/doc-2",
            "subfolder/doc-2",
        ]
    else:  # simple case
        assert len(list(out_ds.get_fragments())) == 1
        assert outs == [
            "subfolder/doc-1",
            "subfolder/doc-2",
            "subfolder/doc-2",
            "subfolder/doc-2",
            "subfolder/doc-2",
            "subfolder/doc-3",
        ]
