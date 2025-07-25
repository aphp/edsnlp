from edsnlp.data.converters import MarkupToDocConverter
from edsnlp.pipes.misc.explode import Explode
from edsnlp.utils.span_getters import get_spans_with_group


def make_doc():
    """
    Build a single document with two entities (entity, date) in `doc.ents`
    and one adjective span in the custom span-group `adj`.
    """
    converter = MarkupToDocConverter(
        preset="xml",
        span_setter={"ents": ["entity", "date"], "adj": ["adj"]},
    )
    return converter(
        "Ceci est un <entity>texte</entity> très <adj>important</adj>, "
        "écrit le <date is_recent>25 juil. 2025</date>"
    )


def groups_in(doc):
    """
    Return the set of non-empty span-group names contained in `doc`.
    (`ents` is treated as a group of its own.)
    """
    groups = set()
    if len(doc.ents):
        groups.add("ents")
    groups.update(name for name, spans in doc.spans.items() if len(spans))
    return groups


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
def test_explode_multiple_groups():
    """
    • Exploding on both ``ents`` and ``adj`` must yield exactly three docs.
    • Each exploded doc must keep the full text, but contain **one** and only
      one non-empty span-group.
    • The surviving span must be the right one.
    """
    doc = make_doc()
    exploder = Explode(span_getter=["ents", "adj"])

    exploded = list(exploder(doc))
    assert len(exploded) == 3

    # In any order, we expect the three following "profiles"
    profiles = sorted(
        [
            (
                tuple(e.text for e in d.ents),
                [e.text for e in d.spans.get("adj", [])],
            )
            for d in exploded
        ]
    )

    assert profiles == sorted(
        [
            (("texte",), []),
            (("25 juil. 2025",), []),
            ((), ["important"]),
        ]
    )

    # Check that attributes are correctly preserved
    assert exploded[1].ents[0]._.is_recent is True
    assert not exploded[0].ents[0]._.is_recent

    # Each sub-document must contain *exactly* one non-empty span-group
    assert all(len(groups_in(sub)) == 1 for sub in exploded)

    # Original doc remains unaltered
    assert groups_in(doc) == {"ents", "adj"}


def test_explode_single_group():
    """
    Exploding only on ``ents`` yields one doc per entity and leaves
    the ``adj`` group untouched in the originals.
    """
    doc = make_doc()
    exploder = Explode(span_getter="ents")

    exploded = list(exploder(doc))
    assert len(exploded) == 2
    assert all(len(sub.ents) == 1 for sub in exploded)
    assert sorted(e.text for sub in exploded for e in sub.ents) == [
        "25 juil. 2025",
        "texte",
    ]


def test_explode_filter_expr():
    doc = make_doc()
    exploder = Explode(
        span_getter=["ents", "adj"],
        filter_expr="any('t' in e.text for e in (*doc.ents, *doc.spans['adj']))",
    )

    exploded = list(exploder(doc))
    assert len(exploded) == 2
    ents = [
        (e.text, g)
        for sub in exploded
        for e, g in get_spans_with_group(sub, exploder.span_getter)
    ]
    assert ents == [("texte", "ents"), ("important", "adj")]


def test_explode_no_spans():
    """
    When the document has no spans selected by the getter, the original
    doc must be yielded unchanged (identity check included).
    """
    import spacy

    nlp = spacy.blank("fr")
    doc = nlp("Pas d'entités ici.")
    exploder = Explode(span_getter="ents")

    result = list(exploder(doc))
    assert result == [doc]  # same object
    assert len(result[0].ents) == 0
