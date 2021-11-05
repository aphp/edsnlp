from typing import List, Tuple

# Source : https://util.unicode.org/UnicodeJsps/character.jsp?a=02EE
quotes: List[str] = [
    "＂",
    "〃",
    "ײ",
    "᳓",
    "″",
    "״",
    "‶",
    "˶",
    "ʺ",
    "“",
    "”",
    "˝",
    "‟",
]

# Source : https://util.unicode.org/UnicodeJsps/character.jsp?a=0027
apostrophes: List[str] = [
    "｀",
    "΄",
    "＇",
    "ˈ",
    "ˊ",
    "ᑊ",
    "ˋ",
    "ꞌ",
    "ᛌ",
    "𖽒",
    "𖽑",
    "‘",
    "’",
    "י",
    "՚",
    "‛",
    "՝",
    "`",
    "`",
    "′",
    "׳",
    "´",
    "ʹ",
    "˴",
    "ߴ",
    "‵",
    "ߵ",
    "ʹ",
    "ʻ",
    "ʼ",
    "´",
    "᾽",
    "ʽ",
    "῾",
    "ʾ",
    "᾿",
]

quotes_and_apostrophes: List[Tuple[str, str]] = [
    ("".join(quotes), '"'),
    ("".join(apostrophes), "'"),
]
