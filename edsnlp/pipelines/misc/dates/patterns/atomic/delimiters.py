from edsnlp.utils.regex import make_pattern

raw_delimiters = [r"\/", r"[-−]"]
delimiters = raw_delimiters + [r"\.", r"[^\S]+"]

raw_delimiter_pattern = make_pattern(raw_delimiters)
raw_delimiter_with_spaces_pattern = make_pattern(raw_delimiters + [r"[^\S]+"])
delimiter_pattern = make_pattern(delimiters)

ante_num_pattern = (
    f"(?<!.(?:{raw_delimiter_pattern})|[.:%a-zA-Z]|[0-9][.:%][ ]?|[0-9][,]?)"
)
post_num_pattern = f"(?!{raw_delimiter_pattern}|[%a-zA-Z]|[ ]?[.:%][0-9]|[.,:]?[0-9])"

ante_num_with_letter_pattern = "(?<!/|[.:%]|[0-9][-−.:%][ ]?|[0-9][,]?)"
post_num_with_letter_pattern = "(?!/|[%]|[ ]?[-−.:%][0-9]|[.,:]?[0-9])"
