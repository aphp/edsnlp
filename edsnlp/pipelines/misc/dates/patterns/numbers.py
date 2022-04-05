from edsnlp.utils.regex import make_pattern

letter_numbers = [
    r"(?P<number_01>l'|le|la|une?|ce|cette|cet)",
    r"(?P<number_02>deux)",
    r"(?P<number_03>trois)",
    r"(?P<number_04>quatre)",
    r"(?P<number_05>cinq)",
    r"(?P<number_06>six)",
    r"(?P<number_07>sept)",
    r"(?P<number_08>huit)",
    r"(?P<number_09>neuf)",
    r"(?P<number_10>dix)",
    r"(?P<number_11>onze)",
    r"(?P<number_12>douze)",
    r"(?P<number_12>treize)",
    r"(?P<number_13>quatorze)",
    r"(?P<number_14>quinze)",
    r"(?P<number_15>seize)",
    r"(?P<number_16>dix[-\s]sept)",
    r"(?P<number_17>dix[-\s]huit)",
    r"(?P<number_18>dix[-\s]neuf)",
    r"(?P<number_20>vingt)",
    r"(?P<number_21>vingt[-\s]et[-\s]un)",
    r"(?P<number_22>vingt[-\s]deux)",
    r"(?P<number_23>vingt[-\s]trois)",
    r"(?P<number_24>vingt[-\s]quatre)",
    r"(?P<number_25>vingt[-\s]cinq)",
    r"(?P<number_26>vingt[-\s]six)",
    r"(?P<number_27>vingt[-\s]sept)",
    r"(?P<number_28>vingt[-\s]huit)",
    r"(?P<number_29>vingt[-\s]neuf)",
    r"(?P<number_30>trente)",
]

numeric_numbers = [str(i) for i in range(1, 100)]

letter_number_pattern = make_pattern(letter_numbers, with_breaks=True)
numeric_number_pattern = make_pattern(numeric_numbers, name="number")

number_pattern = f"({letter_number_pattern}|{numeric_number_pattern})"
