import re

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
