import re
from typing import Dict, Union

from spacy.tokens import Span

extract_start = r".*?[\n\W]*?"
float_regex = r"\d+(?:[,.]\d*)?|[,.]\d+"
int_regex = r"\d+"


def make_find_value_and_reference(admissible_references, default_reference):
    def find_value_and_reference(span):
        possible_values = re.findall(
            rf"((?<!/){float_regex})(?:\s*/\s*({int_regex}))?", span.text
        )
        kept_value = None
        kept_reference = None
        to_keep = False

        for value, reference in possible_values:
            val = float(value.replace(",", "."))
            try:
                ref = int(reference)
                if ref in admissible_references and val <= ref:
                    kept_value, kept_reference = val, ref
                    to_keep = True
                    break
            except ValueError:
                if (kept_reference is None) and (
                    (kept_value is None) or (kept_value < val)
                ):
                    kept_value, kept_reference = (val, None)

        if kept_value is None:
            return

        if (kept_reference is None) and (kept_value <= default_reference):
            to_keep = True
            kept_reference = default_reference

        if to_keep:
            span._.assigned["value"] = kept_value
            span._.assigned["reference"] = kept_reference
            return span

    return find_value_and_reference


def severity_assigner_equals_reference(ent: Span):
    value = ent._.assigned.get("value", None)
    reference = ent._.assigned.get("reference", None)
    assert value is not None, "ent should have a value not None set in _.assigned"
    assert reference is not None, (
        "ent should have a reference not None set in _.assigned"
    )
    if value == reference:
        return "healthy"
    else:
        return "altered_nondescript"


def make_severity_assigner_threshold(
    threshold: Union[int, Dict[int, int]],
    healthy: str = "high",
    comparison: str = "lax",
):
    # preprocessing threshold here because of weird referencing error
    if isinstance(threshold, dict):

        def pre_processing_threshold(ent: Span):
            reference = ent._.assigned.get("reference", None)
            assert reference is not None, (
                "ent should have a reference not None set in _.assigned"
            )
            current_threshold = threshold[reference]
            return current_threshold

    else:

        def pre_processing_threshold(ent: Span):
            return threshold

    def severity_assigner(ent: Span):
        current_threshold = pre_processing_threshold(ent)

        value = ent._.assigned.get("value", None)
        assert value is not None, "ent should have a value not None set in _.assigned"
        if comparison == "lax":
            threshold_comparison = (
                value >= current_threshold
                if healthy == "high"
                else value <= current_threshold
            )
        else:
            threshold_comparison = (
                value > current_threshold
                if healthy == "high"
                else value < current_threshold
            )
        if threshold_comparison:
            return "healthy"
        else:
            return "altered_nondescript"

    return severity_assigner
