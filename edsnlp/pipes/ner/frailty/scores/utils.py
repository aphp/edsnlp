import re

extract_start = r".*?[\n\W]*?"
float_regex = r"\d+(?:[,.]\d*)?|[,.]\d+"
int_regex = r"\d+"

# We reject a value attached to a letter, a sign or another number, e.g. `G6`, `-2`
# or the `5` of `7.5`, a value after a slash or `sur` unless it has its own
# reference, e.g. `10` in `EVA sur 10 : 0` but not `0` in `IADL sur 0/8`, and the
# items of a sum, e.g. `4+4+3`. Since a slash left after the value and its
# reference means a date or a malformed score, e.g. `3/7.5`, we reject those too
score_value = (
    rf"(?<![\w/.,+-])(?<!/\s)(?:(?<!\bsur\s)|(?=\d+\s*/))"
    rf"(?:{float_regex})(?![.,]?\w|\s*\+)"
)
score_reference = rf"\s*(?:/|\bsur\b)\s*{int_regex}(?![.,]?\w)"
# Assign regex capturing the value, the reference stays in the full match
assigned_score_value = rf"({score_value})(?:{score_reference})?(?!\s*/)"
# Normalization regex capturing the value and the optional reference
value_and_reference = re.compile(
    rf"({score_value})(?:\s*(?:/|\bsur\b)\s*({int_regex})(?![.,]?\w))?(?!\s*/)"
)


def make_linked_score_value(name: str) -> str:
    """
    Build the assign regex of a score whose name is also an ordinary word or
    another abbreviation, e.g. `en bref, 3 jours` or `GDS veineux 7.40` for blood
    gases. A value with its reference, e.g. `15/18`, can appear anywhere in the
    window but a value alone must follow the score name, possibly after `:`, `=`,
    `score`, `grade`, `est`, `a` or `de`
    """
    return (
        rf"(?:(?<=\b(?:{name})[\s:=]*(?:(?:score|grade|est|a|de)\b[\s:=]*)*)"
        rf"|(?={score_value}\s*/)){assigned_score_value}"
    )


# We reject a number followed by another unit, e.g. `4 m` or `70 kg`
other_unit = r"(?![.,]?\w)(?!\s*(?:/|%|(?:[ck]?m|kg)\b))"
# Assign regex capturing a duration, e.g. `15s`, `25 secondes`, `2 min` or
# `1 min 20 s`
assigned_duration = (
    rf"(?<![\w/.,+-])((?:{float_regex})\s*(?:(?:min(?:ute)?s?|mn)"
    rf"(?:\s*{int_regex}\s*(?:s|sec(?:onde)?s?)?)?|s|sec(?:onde)?s?)?){other_unit}"
)


def find_duration(span):
    """
    Convert the duration assigned to timed up and go or chair stand scores to
    seconds, durations above 200 seconds are rejected
    """
    # When no duration follows the trigger there is no value, so we reject the match
    text = span._.assigned.get("value")
    if text is None:
        return None
    value, minutes, seconds = re.match(
        rf"({float_regex})\s*(?:(min|mn)\D*(\d+)?)?", text
    ).groups()
    value = float(value.replace(",", "."))
    if minutes is not None:
        value = value * 60 + int(seconds or 0)
    if value <= 200:
        span._.assigned["value"] = value
        return span


# Limit assign stopping the value search of a score at the next score name
score_limit = dict(
    name="limit_scores",
    regex=(
        r"(\b(?:i?adl|mmse?|moca|bref|mini[\s-]+(?:gds|cog)|gds|g8|oncodage|"
        r"(?:ecog[\s-]?)?ps|oms|ecog|rockwood|sppb|tg?ug|mna|gir)\b)"
    ),
    window=(1, 35),
)


def make_find_value_and_reference(admissible_references, default_reference):
    def find_value_and_reference(span):
        possible_values = value_and_reference.findall(span.text)
        # Compact scores such as `ECOG0` have no standalone number, so we fall back on
        # the value captured by their pattern
        if not possible_values and span._.assigned.get("value"):
            value = re.search(float_regex, span._.assigned["value"]).group()
            possible_values = [(value, "")]
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
