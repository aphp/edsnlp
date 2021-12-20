hour_pattern = r"(?<!\d)(?P<hour>0?[1-9]|1\d|2[0-3])(?!\d)"
hour_pattern_with_leading_zero = r"(?<!\d)(?P<hour>0[1-9]|[12]\d|3[01])(?!\d)"

minute_pattern = r"(?<!\d)(?P<minute>0?[1-9]|[1-5]\d)(?!\d)"
minute_pattern_with_leading_zero = r"(?<!\d)(?P<minute>0[1-9]|[1-5]\d)(?!\d)"

second_pattern = r"(?<!\d)(?P<second>0?[1-9]|[1-5]\d)(?!\d)"
second_pattern_with_leading_zero = r"(?<!\d)(?P<second>0[1-9]|[1-5]\d)(?!\d)"

time_pattern = (
    r"(\s.{,3}"
    + f"{hour_pattern}[h:]({minute_pattern_with_leading_zero})?((:|m|min){second_pattern_with_leading_zero})?"
    + ")?"
)
