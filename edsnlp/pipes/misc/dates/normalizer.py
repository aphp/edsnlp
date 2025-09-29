import string
from typing import List, cast

import regex
from spacy.tokens import Span

from edsnlp import registry
from edsnlp.core import PipelineProtocol
from edsnlp.pipes.base import BaseComponent
from edsnlp.pipes.misc.dates.models import AbsoluteDate
from edsnlp.utils.span_getters import SpanGetterArg, get_spans, validate_span_getter

noun_regex = r"""(?xi)
# No letter before
(?<![[:alpha:]])
(?:
    # Either a day of week
    (?P<day_of_week>
        lundi|
        mardi|
        mercredi|
        jeudi|
        vendredi|
        samedi|
        dimanche
    )|
    # Or a month
    (?P<month_1>jan[[:alpha:]]*)|
    (?P<month_2>fev[[:alpha:]]*|fév[[:alpha:]]*)|
    (?P<month_3>mar[[:alpha:]]*)|
    (?P<month_4>avr[[:alpha:]]*)|
    (?P<month_5>mai)|
    (?P<month_6>jun[[:alpha:]]*|juin[[:alpha:]]*)|
    (?P<month_7>jul[[:alpha:]]*|juil[[:alpha:]]*)|
    (?P<month_8>aou[[:alpha:]]*|aoû[[:alpha:]]*)|
    (?P<month_9>sep[[:alpha:]]*)|
    (?P<month_10>oct[[:alpha:]]*)|
    (?P<month_11>nov[[:alpha:]]*)|
    (?P<month_12>dec[[:alpha:]]*|déc[[:alpha:]]*)
)
"""

# no delimiter with space between numbers, ex: d d m m y y y y
full_regex = r"""(?x)
^
    (?P<year>(?:19|2[0Û])[\dÛ][\dÛ])
    [.-/](?P<month>[0Û][1-9]|1[Û012])
    [.-/](?P<day>[0Û][1-9]|[12][\dÛ]|3[Û01])
|
    (?P<day>[0Û][1-9]|[12][\dÛ]|3[Û01])
    [.-/]?(?P<month>[0Û][1-9]|1[Û012])
    [.-/]?(?P<year>(?:19|2[0Û])?[\dÛ][\dÛ])
|
    (?P<day>[0Û]\ +[1-9]|[12]\ +[\dÛ]|3\ +[Û01])
    \ *[.-/]?\ *(?<!\d)(?P<month>[0Û]\ +[1-9]|1\ +[Û012])
    \ *[.-/]?\ *(?<!\d)(?P<year>(?:1\ +9|2\ +[0Û])?\ +[\dÛ]\ +[\dÛ])
"""

number_regex = r"""(?x)
# Numbers with digits
[Û\d]+|
# Numbers with letters: we will sum the value of the matched groups (_{value})
(?i:
    (?<![[:alpha:]])
    (?:
        (?P<_1900>mil[[:alpha:]]*[-\s]?neuf[-\s]?cents?)|
        (?P<_2000>deux[-\s]?mil[[:alpha:]]*)
    )?[-\s]?
    (?:
        (?P<_20>vingt-?\s*et|vingt)|
        (?P<_30>trente-?\s*et|trente)|
        (?P<_40>quarante-?\s*et|quarante)|
        (?P<_50>cinquante-?\s*et|cinquante)|
        (?P<_60>soixante-?\s*et|soixante)|
        (?P<_80>quatres?-?\s*vingts?-?\s*et|quatres?-?\s*vingts?)
    )?[-\s]?
    (?:
        (?P<_1>premier|1\s*er|un)|
        (?P<_2>deux)|
        (?P<_3>trois)|
        (?P<_4>quatre)|
        (?P<_5>cinq)|
        (?P<_6>six)|
        (?P<_7>sept)|
        (?P<_8>huit)|
        (?P<_9>neuf)|
        (?P<_17>dix-?\s*sept)|
        (?P<_18>dix-?\s*huit)|
        (?P<_19>dix-?\s*neuf)|
        (?P<_10>dix)|
        (?P<_11>onze)|
        (?P<_12>douze)|
        (?P<_13>treize)|
        (?P<_14>quatorze)|
        (?P<_15>quinze)|
        (?P<_16>seize)
    )
)
"""


@registry.factory.register(
    "eds.dates_normalizer",
    requires=["doc.ents", "doc.spans"],
    assigns=["token._.date", "token._.date_format"],
)
class DatesNormalizer(BaseComponent):
    def __init__(
        self,
        nlp: PipelineProtocol,
        name: str = "dates_normalizer",
        span_getter: SpanGetterArg = {"ents": True},
    ):
        super().__init__(nlp=nlp, name=name)
        self.span_getter = validate_span_getter(span_getter)

    def set_extensions(self):
        if not Span.has_extension("date"):
            Span.set_extension("date", default=None)
        if not Span.has_extension("datetime"):
            Span.set_extension("datetime", default=None)
        if not Span.has_extension("date_format"):
            Span.set_extension("date_format", default=None)

    @staticmethod
    def extract_date(s, next_date=None, next_offsets=None):
        date_conf = {}

        m = regex.search(full_regex, s)
        if m:
            date = AbsoluteDate(
                day=int(m.group("day").replace(" ", "").replace("Û", "0")),
                month=int(m.group("month").replace(" ", "").replace("Û", "0")),
                year=int(m.group("year").replace(" ", "").replace("Û", "0")),
            )
            return date, sorted(
                [
                    (m.start("day"), m.end("day"), "d"),
                    (m.start("month"), m.end("month"), "m"),
                    (m.start("year"), m.end("year"), "y"),
                ]
            )

        matches = []
        remaining = []

        for match in regex.finditer(noun_regex, s):
            if match.group("day_of_week"):
                remaining.append((match.start(), match.end(), "w", None))
            else:
                value = next(m for m, value in match.groupdict().items() if value)
                matches.append((match.start(), match.end(), "m", int(value[6:])))

        numbers = list(regex.finditer(number_regex, s))
        for match in numbers:
            snippet = match.group()
            if snippet.replace("Û", "0").isdigit():
                value = int(snippet.replace("Û", "0"))
            else:
                value = sum(int(m[1:]) for m, v in match.groupdict().items() if v)
            if value == 0:
                matches.append((match.start(), match.end(), ".", None))
            elif value <= 12:
                if len(snippet) == 1:
                    matches.append((match.start(), match.end(), "dm", value))
                else:
                    matches.append((match.start(), match.end(), "dmy", value))
            elif value <= 31:
                matches.append((match.start(), match.end(), "dy", value))
            elif value <= 40:
                matches.append((match.start(), match.end(), "y", 2000 + value))
            elif 40 < value < 100:
                matches.append((match.start(), match.end(), "y", 1900 + value))
            elif 1900 <= value <= 2100:
                matches.append((match.start(), match.end(), "y", value))
            elif 1900 <= int(snippet[:4]) <= 2100:
                value = int(snippet[:4])
                matches.append((match.start(), match.start() + 4, "y", value))
            else:
                matches.append((match.start(), match.start() + 4, ".", value))

        matches = sorted(matches)

        pattern: List[str] = [m[2] for m in matches]  # type: ignore

        last = -1
        found = "".join(p for p in pattern if len(p) == 1)
        while len(found) != last:
            pattern = [(p.strip(found)) if len(p) > 1 else p for p in pattern]
            last = len(found)
            found = "".join(p for p in pattern if len(p) == 1)

        # In France, dates follow the d/m/y order
        if len(pattern) >= 2 and "d" in pattern[0] and "m" in pattern[1]:
            remaining = ["." if p in ("d", "m") else p for p in remaining]
            pattern = ["." if p in ("d", "m") else p for p in pattern]
            pattern[0] = "d"
            pattern[1] = "m"
        if len(pattern) >= 2 and "m" in pattern[-2] and "y" in pattern[-1]:
            remaining = ["." if p in ("m", "y") else p for p in remaining]
            pattern = ["." if p in ("m", "y") else p for p in pattern]
            pattern[-2] = "m"
            pattern[-1] = "y"
        if len(pattern) >= 3 and "y" in pattern[0] and "m" in pattern[1]:
            remaining = ["." if p in ("d", "m", "y") else p for p in remaining]
            pattern = ["." if p in ("d", "m", "y") else p for p in pattern]
            pattern[0] = "y"
            pattern[1] = "m"
            pattern[2] = "d"

        # Handle cases like [10] in "du [10] au [12/08/1995]"
        if (
            pattern
            and next_date is not None
            and "m" not in pattern
            and "y" not in pattern
            and len(["d" in p for p in pattern]) == 1
            and next_date.month is not None
            and next_date.day is not None
            and next_offsets[0][2] not in "ym"
        ):
            if next_date.year is not None:
                date_conf["year"] = next_date.year
            date_conf["month"] = next_date.month
            found += "ym"
            pattern = ["d" if "d" in p else p for p in pattern]
        # Handle cases like [10] in "de [mai] à [juin 1996]"
        elif (
            pattern
            and next_date is not None
            and "y" not in pattern
            and len(["m" in p for p in pattern]) == 1
            and next_date.year is not None
            and next_date.month is not None
            and next_offsets[0][2] not in "yd"
        ):
            date_conf["year"] = next_date.year
            found += "y"
            pattern = ["m" if "m" in p else p for p in pattern]

        # Handle year found but missing month => remove day
        forbidden = ""
        found = pattern + [m[2] for m in remaining]
        if "d" in found and "y" in found and "m" not in found:
            forbidden += "d"

        if len(["d" in p for p in pattern]) == 1:
            pattern = ["d" if "d" in p else p for p in pattern]

        # Handle missing day => remove day of week
        if "w" in found and "d" not in found:
            forbidden += "w"

        matches = sorted(
            [
                (m[0], m[1], p if len(p) == 1 and p not in forbidden else ".", m[3])
                for m, p in (
                    *zip(matches, pattern),
                    *zip(remaining, [m[2] for m in remaining]),
                )
            ]
        )

        for m in matches:
            if m[2] == "d":
                date_conf["day"] = m[3]
            elif m[2] == "m":
                date_conf["month"] = m[3]
            elif m[2] == "y":
                date_conf["year"] = m[3]

        date = cast(AbsoluteDate, date_conf)

        return date, [(b, e, k) for b, e, k, v in matches]

    def extract_format(self, s, matches):
        date_format = ""
        offset = 0
        translation = str.maketrans(string.digits, "?" * len(string.digits))
        for begin, end, kind in matches:
            date_format += s[offset:begin].translate(translation).replace("%", "%%")
            snippet = s[begin:end]
            if kind == "d":
                if snippet.isdigit() and len(snippet) == 2:
                    date_format += "%d"
                else:
                    date_format += "%-d"
            elif kind == "m":
                if snippet.isdigit():
                    if len(snippet) == 1:
                        date_format += "%-m"
                    else:
                        date_format += "%m"
                else:
                    if len(snippet) <= 3:
                        date_format += "%b"
                    else:
                        date_format += "%B"
            elif kind == "w":
                if len(snippet) <= 3:
                    date_format += "%a"
                else:
                    date_format += "%A"
            elif kind == "y":
                date_format += "%Y"
            elif kind == ".":
                date_format += snippet.translate(translation).replace("%", "%%")
            offset = end

        date_format += s[offset:].translate(translation).replace("%", "%%")

        return date_format

    def __call__(self, doc):
        spans = list(get_spans(doc, self.span_getter))
        last_date = None
        last_date_offsets = None
        last_span = None
        for span in reversed(spans):
            span: Span
            text = span.text
            if last_span and last_span.start - span.end > 2:
                last_date = last_date_offsets = None

            date, date_offsets = self.extract_date(
                text,
                last_date,
                last_date_offsets,
            )
            span._.date = date
            span._.datetime = date.to_datetime(doc._.note_datetime)
            span._.date_format = self.extract_format(text, date_offsets)

            last_span, last_date, last_date_offsets = span, date, date_offsets

        return doc
