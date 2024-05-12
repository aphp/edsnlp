import warnings
from datetime import date, datetime

from dateutil.parser import parse as parse_date
from spacy.tokens import Doc

if not Doc.has_extension("note_id"):
    Doc.set_extension("note_id", default=None)


def set_note_datetime(doc, dt):
    try:
        if type(dt) is datetime:
            pass
        elif isinstance(dt, str):
            dt = parse_date(dt)
        elif isinstance(dt, (int, float)):
            dt = datetime.fromtimestamp(dt)
        elif isinstance(dt, date):
            dt = datetime(dt.year, dt.month, dt.day)
        elif dt is None:
            pass
        key = doc._._get_key("note_datetime")
        doc.doc.user_data[key] = dt
        return
    except Exception:
        pass

    warnings.warn(f"Cannot cast {dt} as a note datetime", UserWarning)


def get_note_datetime(doc):
    key = doc._._get_key("note_datetime")
    return doc.user_data.get(key, None)


if not Doc.has_extension("note_datetime"):
    Doc.set_extension(
        "note_datetime",
        getter=get_note_datetime,
        setter=set_note_datetime,
    )

if not Doc.has_extension("birth_datetime"):
    Doc.set_extension("birth_datetime", default=None)
