from edsnlp.pipelines.misc.dates.patterns.absolute import absolute_pattern, mode_pattern
from edsnlp.utils.regex import make_pattern

numeric_day_pattern = r"(?<!\d)(0 [1-9]|[12] \d|3 [01])(?!\d)"
numeric_month_pattern = r"(?<!\d)(0 [1-9]|1 [0-2])(?!\d)"

year_patterns = [
    r"1 9 \d \d",
] + [" ".join(str(year)) for year in range(2000, 2024)]
full_year_pattern = make_pattern(year_patterns, name="year")


day_pattern = f"(?P<day>{numeric_day_pattern})"
month_pattern = f"(?P<month>{numeric_month_pattern})"
year_pattern = f"(?P<year>{full_year_pattern})"

pseudo_date_pattern = [
    r"(?<="
    + mode_pattern
    + r".{,3})?"
    + f"{day_pattern} {month_pattern} {year_pattern}"
] + absolute_pattern
