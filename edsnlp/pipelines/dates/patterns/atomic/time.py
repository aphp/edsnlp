hour_pattern = r"(?<!\d)(?P<hour>0?[1-9]|1\d|2[0-3])(?!\d)"
lz_hour_pattern = r"(?<!\d)(?P<hour>0[1-9]|[12]\d|3[01])(?!\d)"

minute_pattern = r"(?<!\d)(?P<minute>0?[1-9]|[1-5]\d)(?!\d)"
lz_minute_pattern = r"(?<!\d)(?P<minute>0[1-9]|[1-5]\d)(?!\d)"

second_pattern = r"(?<!\d)(?P<second>0?[1-9]|[1-5]\d)(?!\d)"
lz_second_pattern = r"(?<!\d)(?P<second>0[1-9]|[1-5]\d)(?!\d)"

# The time pattern is always optional
time_pattern = (
    r"(\s.{,3}"
    + f"{hour_pattern}[h:]({lz_minute_pattern})?"
    + f"((:|m|min){lz_second_pattern})?"
    + ")?"
)
