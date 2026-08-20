import pytest
import regex

import edsnlp
from edsnlp.pipes.ner.tnm.model import TNM
from edsnlp.pipes.ner.tnm.patterns import tnm_pattern
from edsnlp.utils.examples import parse_example
from edsnlp.utils.typing import cast

examples = [
    # Basic full TNM combinations
    "TNM: <ent norm=aTxN1M0>aTxN1M0</ent>",
    "TNM: <ent norm=aTxN1M0>aTxN1M0</ent> ",
    "TNM: <ent norm=cT3N0M0>cT3N0M0</ent> \n \n",
    "TNM: <ent norm=aTxN1M0>aTx / N1 / M0</ent>",
    "TNM: <ent norm=aTxN1R2>aTxN1 R2</ent>",
    "TNM: <ent norm=pT2N1mi>pT2 N1mi</ent>",
    "TNM: <ent norm=pT1mN1M0>pT1(m)N1 M0</ent>",
    "TNM: <ent norm=pT2cN0R0>pT2c N0 R0</ent>",
    "TNM: <ent norm=pT2N0R0>pT2N0R0</ent>",
    "TNM: <ent norm=pT2N1M0R0>pT2N1M0R0</ent>",
    # Space inside M value — regression for the m-spec/M-component conflict
    "TNM: <ent norm=pTxN1M0>p Tx N1M 0</ent>",
    # node_specification with parenthesised forms — parens stripped in norm
    "TNM: <ent norm=pT1bN0sn>pT1bN0(sn)</ent>",
    # node prefix
    "TNM: <ent norm=pT1pN1M0>pT1 pN1 M0</ent>\n \n ",
    # Standalone T: requires prefix AND specification (branch 1 of logic_filter)
    "TNM: <ent norm=pT2b>pT2b</ent>",
    "TNM: <ent norm=yT1a>yT1a</ent>",
    # Case variants — (?i) flag must handle all combos
    "TNM: <ent norm=pT2N1M0>pt2n1m0</ent>",  # all lowercase
    "TNM: <ent norm=PT2N1M0>PT2N1M0</ent>",  # uppercase prefix P
    "TNM: <ent norm=pT2N1M0>pT2N1m0</ent>",  # lowercase m for metastasis letter
    "TNM: <ent norm=aTXN1M0>aTXN1M0</ent>",  # uppercase X tumour value
    # Delimiter variants between components
    "TNM: <ent norm=pT2N1M0>pT2,N1,M0</ent>",
    "TNM: <ent norm=pT2N1M0>pT2/N1/M0</ent>",
    "TNM: <ent norm=pT2N1M0>pT2, N1, M0</ent>",
    # Logic filter — positive: T alone without prefix+spec must have N, M, or R
    "TNM: <ent norm=T2N1>T2N1</ent>",  # no prefix, N present
    "TNM: <ent norm=pT2M0>pT2M0</ent>",  # no N, M present
    "TNM: <ent norm=T2aN1M0>T2aN1M0</ent>",  # no prefix, spec+NMR present
    "TNM: <ent norm=T2R0>T2R0</ent>",  # no prefix, R present
    # Free-text fields must keep their `o` (only stage fields are coerced)
    "TNM: <ent norm=pT2N1mol+M0>pT2N1(mol+)M0</ent>",
    "TNM: <ent norm=pT2N1M1oss>pT2N1M1oss</ent>",
    "TNM: <ent norm=pT4N2R1foie>pT4N2R1(foie)</ent>",
    # Pleural invasion, and a bare `PL` that must not be picked up
    "TNM: <ent norm=pT2N1M0PL1>pT2N1M0 PL1</ent>",
    "TNM: <ent norm=pT2N1M0>pT2N1M0</ent> PL",
    # Per-component prefixes are independent
    "TNM: <ent norm=pT1cN1M0>pT1 cN1 M0</ent>",
    # Node ratio specification
    "TNM: <ent norm=pT2N13/12M0>pT2N1(3/12)M0</ent>",
    # `o`/`O` typed instead of the digit zero
    "TNM: <ent norm=pT0N1M0>pTON1MO</ent>",
    # Classification version: part of the span, and of the normalised value
    "TNM: <ent norm='pT2N1M0 (UICC 2017)'>pT2N1M0 (UICC 2017)</ent>",
    "TNM: <ent norm='pT2N1M0 (UICC 2017)'>pT2N1M0 (UICC 17)</ent>",
    "TNM: <ent norm='pT2N1M0R0 (TNM 2009)'>pT2N1M0R0, TNM 2009</ent>",
    # A parenthesised free text is captured as a suffix but stays out of norm()
    "TNM: <ent norm=pT2N1M0>pT2N1M0 (2017)</ent>",
    # Logic filter — negative
    "TNM: T2a",  # spec present but no prefix and no NMR
    # Should NOT match — logic_filter rejects bare T without N/M/R or prefix+spec
    "TNM: PT",
    "TNM: p    T \n",
    "TNM: a T \n",
    "TNM: T2",
    "TNM: pT2",
]


def test_tnm(blank_nlp):
    blank_nlp.add_pipe("eds.tnm", config=dict(pattern=tnm_pattern))

    for example in examples:
        text, entities = parse_example(example=example)

        doc = blank_nlp(text)

        assert len(entities) == len(doc.ents), (
            f"Expected {len(entities)} entities, got {len(doc.ents)} in {text!r}\n"
            f"  Found: {[e.text for e in doc.ents]}"
        )

        for entity, ent in zip(entities, doc.ents):
            norm = entity.modifiers[0].value
            expected_span = text[entity.start_char : entity.end_char]
            assert ent.text == expected_span, (
                f"Span mismatch in {text!r}: "
                f"expected {expected_span!r}, got {ent.text!r}"
            )
            assert norm == ent._.value.norm(), (
                f"Norm mismatch in {text!r}: "
                f"expected {norm!r}, got {ent._.value.norm()!r}"
            )


# ---------------------------------------------------------------------------
# Decomposition tests — pure regex + model, no pipeline required
# Each entry: (input_text, expected_fields, expected_norm)
# expected_fields covers only the fields we want to assert; absent keys are
# not checked (they may be None or carry irrelevant capture-group artefacts).
# ---------------------------------------------------------------------------

decomposition_cases = [
    # --- prefix variants ---
    (
        "pT2N1M0",
        {"tumour_prefix": "p", "tumour": "2", "node": "1", "metastasis": "0"},
        "pT2N1M0",
    ),
    (
        "cT3N0M0",
        {"tumour_prefix": "c", "tumour": "3", "node": "0", "metastasis": "0"},
        "cT3N0M0",
    ),
    (
        "aTxN1M0",
        {"tumour_prefix": "a", "tumour": "x", "node": "1", "metastasis": "0"},
        "aTxN1M0",
    ),
    (
        "ypT2N1M0",
        {"tumour_prefix": "yp", "tumour": "2", "node": "1", "metastasis": "0"},
        "ypT2N1M0",
    ),
    # --- tumour specification ---
    (
        "pT1bN0M0",
        {
            "tumour_prefix": "p",
            "tumour": "1",
            "tumour_specification": "b",
            "node": "0",
            "metastasis": "0",
        },
        "pT1bN0M0",
    ),
    (
        "pT2cN0M0",
        {
            "tumour_prefix": "p",
            "tumour": "2",
            "tumour_specification": "c",
            "node": "0",
            "metastasis": "0",
        },
        "pT2cN0M0",
    ),
    # --- tumour suffix (multifocal) ---
    (
        "pT1(m)N1M0",
        {
            "tumour_prefix": "p",
            "tumour": "1",
            "tumour_suffix": "m",
            "node": "1",
            "metastasis": "0",
        },
        "pT1mN1M0",
    ),
    # --- node specification (text form) ---
    (
        "pT2N1mi",
        {
            "tumour_prefix": "p",
            "tumour": "2",
            "node": "1",
            "node_specification": "mi",
        },
        "pT2N1mi",
    ),
    (
        "pT1bN0sn",
        {
            "tumour_prefix": "p",
            "tumour": "1",
            "tumour_specification": "b",
            "node": "0",
            "node_specification": "sn",
        },
        "pT1bN0sn",
    ),
    # --- node specification (parenthesised form — raw value keeps parens,
    #     norm() strips them) ---
    (
        "pT1bN0(sn)",
        {
            "tumour_prefix": "p",
            "tumour": "1",
            "tumour_specification": "b",
            "node": "0",
            "node_specification": "(sn)",
        },
        "pT1bN0sn",
    ),
    # --- node prefix ---
    (
        "pT1 pN1 M0",
        {
            "tumour_prefix": "p",
            "tumour": "1",
            "node_prefix": "p",
            "node": "1",
            "metastasis": "0",
        },
        "pT1pN1M0",
    ),
    # --- metastasis specification ---
    (
        "pT2N1M1PUL",
        {
            "tumour_prefix": "p",
            "tumour": "2",
            "node": "1",
            "metastasis": "1",
            "metastasis_specification": "PUL",
        },
        "pT2N1M1PUL",
    ),
    # --- resection ---
    (
        "pT2N0R0",
        {
            "tumour_prefix": "p",
            "tumour": "2",
            "node": "0",
            "resection": "0",
        },
        "pT2N0R0",
    ),
    (
        "pT2N1M0R0",
        {
            "tumour_prefix": "p",
            "tumour": "2",
            "node": "1",
            "metastasis": "0",
            "resection": "0",
        },
        "pT2N1M0R0",
    ),
    # --- standalone T: branch 1 (prefix + spec, no N/M/R) ---
    (
        "pT2b",
        {
            "tumour_prefix": "p",
            "tumour": "2",
            "tumour_specification": "b",
            "node": None,
            "metastasis": None,
        },
        "pT2b",
    ),
    (
        "yT1a",
        {
            "tumour_prefix": "y",
            "tumour": "1",
            "tumour_specification": "a",
            "node": None,
            "metastasis": None,
        },
        "yT1a",
    ),
    # --- regression: space inside M value must not be consumed as node spec ---
    (
        "pT2N1M 0",
        {
            "tumour_prefix": "p",
            "tumour": "2",
            "node": "1",
            "node_specification": None,
            "metastasis": "0",
        },
        "pT2N1M0",
    ),
    # --- case variants: (?i) flag ---
    # All lowercase: raw fields store what the regex captured; norm() rebuilds with
    # hardcoded uppercase T/N/M/R letters.
    (
        "pt2n1m0",
        {
            "tumour_prefix": "p",
            "tumour": "2",
            "node": "1",
            "metastasis": "0",
        },
        "pT2N1M0",
    ),
    (
        "PT2N1M0",
        {
            "tumour_prefix": "P",
            "tumour": "2",
            "node": "1",
            "metastasis": "0",
        },
        "PT2N1M0",
    ),
    # lowercase m for the metastasis letter: node_specification must NOT consume it
    (
        "pT2N1m0",
        {
            "tumour_prefix": "p",
            "tumour": "2",
            "node": "1",
            "node_specification": None,
            "metastasis": "0",
        },
        "pT2N1M0",
    ),
    # uppercase X in tumour value is preserved in raw field and in norm
    (
        "aTXN1M0",
        {
            "tumour_prefix": "a",
            "tumour": "X",
            "node": "1",
            "metastasis": "0",
        },
        "aTXN1M0",
    ),
    # --- delimiter variants ---
    (
        "pT2,N1,M0",
        {
            "tumour_prefix": "p",
            "tumour": "2",
            "node": "1",
            "metastasis": "0",
        },
        "pT2N1M0",
    ),
    (
        "pT2/N1/M0",
        {
            "tumour_prefix": "p",
            "tumour": "2",
            "node": "1",
            "metastasis": "0",
        },
        "pT2N1M0",
    ),
    (
        "pT2, N1, M0",
        {
            "tumour_prefix": "p",
            "tumour": "2",
            "node": "1",
            "metastasis": "0",
        },
        "pT2N1M0",
    ),
    (
        "pT2\nN1\nM0",
        {
            "tumour_prefix": "p",
            "tumour": "2",
            "node": "1",
            "metastasis": "0",
        },
        "pT2N1M0",
    ),
    # --- logic filter positive ---
    # No prefix, N present (branch 2)
    (
        "T2N1",
        {
            "tumour_prefix": None,
            "tumour": "2",
            "node": "1",
            "metastasis": None,
        },
        "T2N1",
    ),
    # No N, M present (branch 2)
    (
        "pT2M0",
        {
            "tumour_prefix": "p",
            "tumour": "2",
            "node": None,
            "metastasis": "0",
        },
        "pT2M0",
    ),
    # No prefix, spec present and NMR present (branch 2 — spec is in addition)
    (
        "T2aN1M0",
        {
            "tumour_prefix": None,
            "tumour": "2",
            "tumour_specification": "a",
            "node": "1",
            "metastasis": "0",
        },
        "T2aN1M0",
    ),
    # No prefix, R present (branch 2)
    (
        "T2R0",
        {
            "tumour_prefix": None,
            "tumour": "2",
            "node": None,
            "metastasis": None,
            "resection": "0",
        },
        "T2R0",
    ),
    # --- `o` coercion is restricted to the numeric stage fields ---
    (
        "pT2N1(mol+)M0",
        {
            "tumour": "2",
            "node": "1",
            "node_specification": "(mol+)",
            "metastasis": "0",
        },
        "pT2N1mol+M0",
    ),
    (
        "pT2N1M1oss",
        {
            "tumour": "2",
            "node": "1",
            "metastasis": "1",
            "metastasis_specification": "oss",
        },
        "pT2N1M1oss",
    ),
    (
        "pT4N2R1(foie)",
        {
            "tumour": "4",
            "node": "2",
            "resection": "1",
            "resection_loc": "(foie)",
        },
        "pT4N2R1foie",
    ),
    # ...but `o`/`O` typed for the digit zero still is coerced
    (
        "pTON1MO",
        {"tumour": "0", "node": "1", "metastasis": "0"},
        "pT0N1M0",
    ),
    # --- pleura ---
    (
        "pT2N1M0 PL1",
        {"tumour": "2", "node": "1", "metastasis": "0", "pleura": "1"},
        "pT2N1M0PL1",
    ),
    # --- node ratio ---
    (
        "pT2N1(3/12)M0",
        {"node": "1", "node_specification": "(3/12)", "metastasis": "0"},
        "pT2N13/12M0",
    ),
    # --- per-component prefixes are independent ---
    (
        "pT1 cN1 M0",
        {
            "tumour_prefix": "p",
            "tumour": "1",
            "node_prefix": "c",
            "node": "1",
            "metastasis": "0",
        },
        "pT1cN1M0",
    ),
    # --- resection specification and free-text suffix ---
    # A free-text suffix is kept in the field but left out of norm()
    (
        "pT2N1M0R0(marge saine)",
        {"resection": "0", "resection_suffix": "marge saine"},
        "pT2N1M0R0",
    ),
    (
        "pT2N1M0 (2017)",
        {"metastasis_suffix": "2017", "version": None, "version_year": None},
        "pT2N1M0",
    ),
    (
        "pT2N0R1is",
        {"resection": "1", "resection_specification": "is"},
        "pT2N0R1is",
    ),
    # --- classification version: captured, not mistaken for a suffix ---
    (
        "pT2N1M0 (UICC 2017)",
        {
            "metastasis": "0",
            "metastasis_suffix": None,
            "version": "UICC",
            "version_year": 2017,
        },
        "pT2N1M0 (UICC 2017)",
    ),
    # Two-digit years are expanded
    (
        "pT2N1M0 (UICC 17)",
        {"version": "UICC", "version_year": 2017},
        "pT2N1M0 (UICC 2017)",
    ),
    (
        "pT2N1M0 uicc 2017",
        {"version": "uicc", "version_year": 2017},
        "pT2N1M0 (UICC 2017)",
    ),
    # Version placed after the resection component
    (
        "pT2N1M0R0 (UICC 2017)",
        {"resection": "0", "version": "UICC", "version_year": 2017},
        "pT2N1M0R0 (UICC 2017)",
    ),
    # Edition wording between the classification name and the year
    (
        "pT2N1M0 (UICC ed. 2017)",
        {"version": "UICC", "version_year": 2017},
        "pT2N1M0 (UICC 2017)",
    ),
    (
        "pT2N1M0 (uicc 7eme edition 2009)",
        {"version": "uicc", "version_year": 2009},
        "pT2N1M0 (UICC 2009)",
    ),
    (
        "pT2N1M0 (UICC 7e ed. 2009)",
        {"version": "UICC", "version_year": 2009},
        "pT2N1M0 (UICC 2009)",
    ),
    # A 4-digit year must not be split by the optional edition ordinal
    (
        "pT2N1M0 (UICC 1987)",
        {"version": "UICC", "version_year": 1987},
        "pT2N1M0 (UICC 1987)",
    ),
    # A 2-digit year of 40 or more belongs to the 20th century
    (
        "pT2N1M0 (UICC 87)",
        {"version": "UICC", "version_year": 1987},
        "pT2N1M0 (UICC 1987)",
    ),
    # --- suffix and prefix on the N, M and R components ---
    (
        "pT2N1mi(cap)M0",
        {"node": "1", "node_specification": "mi", "node_suffix": "cap"},
        "pT2N1micapM0",
    ),
    (
        "pT2N1 pM0",
        {"node": "1", "metastasis_prefix": "p", "metastasis": "0"},
        "pT2N1pM0",
    ),
    (
        "pT2N1M0 pR0",
        {"metastasis": "0", "resection_prefix": "p", "resection": "0"},
        "pT2N1M0pR0",
    ),
    (
        "pT2N1M0 (TNM 2009)",
        {"version": "TNM", "version_year": 2009},
        "pT2N1M0 (TNM 2009)",
    ),
    # No year: neither a version nor a metastasis suffix -- the span stops at M0
    (
        "pT2N1M0 (AJCC 8)",
        {"version": None, "version_year": None, "metastasis_suffix": None},
        "pT2N1M0",
    ),
]


# Logic filter — inputs that must produce NO match
no_match_cases = [
    ("T2a", "spec only, no prefix, no NMR"),
    ("T2", "bare T, no prefix, no spec, no NMR"),
    ("pT2", "prefix but no spec and no NMR"),
    ("PT", "prefix + T but no tumour value"),
    ("N1M0", "the T component is mandatory"),
    ("pN1", "the T component is mandatory"),
    # Stage values are a closed set: validation lives in the pattern, not in
    # the model, so an out-of-range value invalidates the whole mention.
    ("pT7N1M0", "T is limited to 0-4, is, x"),
    ("pT2N5M0", "N is limited to 0-4, x, +"),
    ("pT2N50M0", "N is limited to a single character"),
    ("pT2N1M4", "M is limited to 0-3, x, +"),
    ("pT2N1M0PL4", "PL is limited to 0-3, x"),
    ("pT2N1M0R5", "R is limited to 0-2, x, +"),
]


def _parse(text: str) -> TNM:
    m = regex.search(tnm_pattern, text)
    assert m is not None, f"Pattern did not match {text!r}"
    return cast(TNM, m.groupdict())


def test_tnm_decomposition():
    for text, expected_fields, expected_norm in decomposition_cases:
        tnm = _parse(text)

        for field, expected_value in expected_fields.items():
            actual = getattr(tnm, field)
            assert actual == expected_value, (
                f"[{text!r}] field {field!r}: "
                f"expected {expected_value!r}, got {actual!r}"
            )

        assert tnm.norm() == expected_norm, (
            f"[{text!r}] norm: expected {expected_norm!r}, got {tnm.norm()!r}"
        )


def test_tnm_no_match():
    for text, reason in no_match_cases:
        m = regex.search(tnm_pattern, text)
        assert m is None, (
            f"Pattern should NOT match {text!r} ({reason}), but got {m.group()!r}"
        )


# ---------------------------------------------------------------------------
# banned_words post-filter
# ---------------------------------------------------------------------------


def test_tnm_banned_words():
    """Lookalike abbreviations are dropped, and the list is configurable."""
    nlp = edsnlp.blank("eds")
    nlp.add_pipe("eds.tnm")

    # These would otherwise be matched: `atom` reads as aT0M, `mtxx` as mTx+x,
    # `autonom` as auT0N0m, ...
    for text in ["atom", "autoa", "autonom", "mtxd", "mtxx", "tissunom"]:
        assert not nlp(text).ents, f"{text!r} should be filtered out"

    # Emptying the list lets the raw regex matches through again
    permissive = edsnlp.blank("eds")
    permissive.add_pipe("eds.tnm", config=dict(banned_words=[]))
    assert permissive("mtxx").ents


# ---------------------------------------------------------------------------
# TNM model API
# ---------------------------------------------------------------------------


def test_tnm_model_api():
    """`str()`, `dict()` and the normalisation helpers."""
    tnm = _parse("pT2N1M0")

    assert str(tnm) == tnm.norm() == "pT2N1M0"

    d = tnm.dict()
    assert d["tumour_prefix"] == "p"
    assert d["tumour"] == "2"
    assert d["pleura"] is None
    assert set(d) == set(TNM.model_fields)

    assert tnm.dict(exclude_none=True) == {
        "tumour_prefix": "p",
        "tumour": "2",
        "node": "1",
        "metastasis": "0",
    }

    # Empty values normalise to an empty string rather than raising
    assert TNM._norm_str(None) == ""
    assert TNM._norm_suffix(None) == ""
    # A suffix is only a qualifier when it is one to three letters
    assert TNM._norm_suffix("(m)") == "m"
    assert TNM._norm_suffix("(grade 2)") == ""


def test_tnm_dict_skip_defaults_is_deprecated():
    tnm = _parse("pT2N1M0")

    with pytest.deprecated_call():
        d = tnm.dict(skip_defaults=True)

    assert d["tumour"] == "2"
