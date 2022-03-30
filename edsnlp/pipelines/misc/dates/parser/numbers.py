numbers = {
    1: ["un", "une"],
    2: "deux",
    3: "trois",
    4: "quatre",
    5: "cinq",
    6: "six",
    7: "sept",
    8: "huit",
    9: "neuf",
    10: "dix",
    11: "onze",
    12: "douze",
    13: "treize",
    14: "quatorze",
    15: "quinze",
    16: "seize",
    17: "dix-sept",
    18: "dix-huit",
    19: "dix-neuf",
    20: "vingt",
    21: "vingt-et-un",
    22: "vingt-deux",
    23: "vingt-trois",
    24: "vingt-quatre",
    25: "vingt-cinq",
    26: "vingt-six",
    27: "vingt-sept",
    28: "vingt-huit",
    29: "vingt-neuf",
    30: "trente",
}


letter_numbers = dict()

for k, v in numbers.items():
    if isinstance(v, str):
        v = [v]
    for text in v:
        letter_numbers[text] = k
