import regex

from edsnlp.pipes.ner.tnm.model import TNM
from edsnlp.pipes.ner.tnm.patterns_new import tnm_pattern_new
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
    blank_nlp.add_pipe("eds.tnm", config=dict(pattern=tnm_pattern_new))

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
]


# Logic filter — inputs that must produce NO match
no_match_cases = [
    ("T2a", "spec only, no prefix, no NMR"),
    ("T2", "bare T, no prefix, no spec, no NMR"),
    ("pT2", "prefix but no spec and no NMR"),
    ("PT", "prefix + T but no tumour value"),
]


def _parse(text: str) -> TNM:
    m = regex.search(tnm_pattern_new, text)
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
        m = regex.search(tnm_pattern_new, text)
        assert m is None, (
            f"Pattern should NOT match {text!r} ({reason}), but got {m.group()!r}"
        )
