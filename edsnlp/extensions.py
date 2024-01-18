from datetime import date, datetime

import pendulum
from spacy.tokens import Doc

if not Doc.has_extension("note_id"):
    Doc.set_extension("note_id", default=None)


def set_note_datetime(span, dt):
    if type(dt) is datetime:
        dt = pendulum.instance(dt)
    elif isinstance(dt, pendulum.DateTime):
        pass
    elif isinstance(dt, str):
        dt = pendulum.parse(dt)
    elif isinstance(dt, (int, float)):
        dt = pendulum.from_timestamp(dt)
    elif isinstance(dt, date):
        dt = pendulum.instance(datetime.fromordinal(dt.toordinal()))
    elif dt is None:
        pass
    else:
        raise ValueError(f"Cannot cast {dt} as a datetime")
    key = span._._get_key("note_datetime")
    span.doc.user_data[key] = dt


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
