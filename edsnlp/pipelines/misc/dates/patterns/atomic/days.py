from edsnlp.utils.regex import make_pattern

letter_days = [
    r"(?P<day_01>premier|1\s*er)",
    r"(?P<day_02>deux)",
    r"(?P<day_03>trois)",
    r"(?P<day_04>quatre)",
    r"(?P<day_05>cinq)",
    r"(?P<day_06>six)",
    r"(?P<day_07>sept)",
    r"(?P<day_08>huit)",
    r"(?P<day_09>neuf)",
    r"(?P<day_10>dix)",
    r"(?P<day_11>onze)",
    r"(?P<day_12>douze)",
    r"(?P<day_13>treize)",
    r"(?P<day_14>quatorze)",
    r"(?P<day_15>quinze)",
    r"(?P<day_16>seize)",
    r"(?P<day_17>dix\-?\s*sept)",
    r"(?P<day_18>dix\-?\s*huit)",
    r"(?P<day_19>dix\-?\s*neuf)",
    r"(?P<day_20>vingt)",
    r"(?P<day_21>vingt\-?\s*et\-?\s*un)",
    r"(?P<day_22>vingt\-?\s*deux)",
    r"(?P<day_23>vingt\-?\s*trois)",
    r"(?P<day_24>vingt\-?\s*quatre)",
    r"(?P<day_25>vingt\-?\s*cinq)",
    r"(?P<day_26>vingt\-?\s*six)",
    r"(?P<day_27>vingt\-?\s*sept)",
    r"(?P<day_28>vingt\-?\s*huit)",
    r"(?P<day_29>vingt\-?\s*neuf)",
    r"(?P<day_30>trente)",
    r"(?P<day_31>trente\-?\s*et\-?\s*un)",
]

letter_day_pattern = make_pattern(letter_days)

numeric_day_pattern = r"(?<!\d)(0?[1-9]|[12]\d|3[01])(?!\d)"
lz_numeric_day_pattern = r"(?<!\d)(0[1-9]|[12]\d|3[01])(?!\d)"
nlz_numeric_day_pattern = r"(?<!\d)([1-9]|[12]\d|3[01])(?!\d)"

numeric_day_pattern = f"(?P<day>{numeric_day_pattern})"
lz_numeric_day_pattern = f"(?P<day>{lz_numeric_day_pattern})"

day_pattern = f"({letter_day_pattern}|{numeric_day_pattern})"
