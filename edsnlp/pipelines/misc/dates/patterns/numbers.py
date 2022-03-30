from edsnlp.utils.regex import make_pattern

letter_numbers = {
    r"(l'|le|la|une?)": 1,
    r"deux": 2,
    r"trois": 3,
    r"quatre": 4,
    r"cinq": 5,
    r"six": 6,
    r"sept": 7,
    r"huit": 8,
    r"neuf": 9,
    r"dix": 10,
    r"onze": 11,
    r"douze": 12,
    r"treize": 13,
    r"quatorze": 14,
    r"quinze": 15,
    r"seize": 16,
    r"dix[-\s]sept": 17,
    r"dix[-\s]huit": 18,
    r"dix[-\s]neuf": 19,
    r"vingt": 20,
    r"vingt[-\s]et[-\s]un": 21,
    r"vingt[-\s]deux": 22,
    r"vingt[-\s]trois": 23,
    r"vingt[-\s]quatre": 24,
    r"vingt[-\s]cinq": 25,
    r"vingt[-\s]six": 26,
    r"vingt[-\s]sept": 27,
    r"vingt[-\s]huit": 28,
    r"vingt[-\s]neuf": 29,
    r"trente": 30,
}

numeric_numbers = [str(i) for i in range(1, 100)]

numbers = list(letter_numbers.keys()) + numeric_numbers

year_pattern = make_pattern(numbers, name="year")
month_pattern = make_pattern(numbers, name="month")
week_pattern = make_pattern(numbers, name="week")
day_pattern = make_pattern(numbers, name="day")
hour_pattern = make_pattern(numbers, name="day")
