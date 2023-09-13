from pydantic import validate_arguments

from edsnlp.pipelines.base import (
    SpanGetterArg,
    SpanSetterArg,
    validate_span_getter,
    validate_span_setter,
)


def test_span_getter():
    assert validate_span_getter("ents") == {"ents": True}
    assert validate_span_getter(["ents"]) == {"ents": True}
    assert validate_span_getter(["ents", "group"]) == {"ents": True, "group": True}
    assert validate_span_getter({"grp": True}) == {"grp": True}
    assert validate_span_getter({"grp": ["a", "b", "c"]}) == {"grp": ["a", "b", "c"]}


def test_span_setter():
    assert validate_span_setter("ents") == {"ents": True}
    assert validate_span_setter(["ents"]) == {"ents": True}
    assert validate_span_setter(["ents", "group"]) == {"ents": True, "group": True}
    assert validate_span_setter({"grp": True}) == {"grp": True}
    assert validate_span_setter({"grp": ["a", "b", "c"]}) == {"grp": ["a", "b", "c"]}


def test_validate_args():
    @validate_arguments
    def my_func(span_getter: SpanGetterArg, span_setter: SpanSetterArg):
        return span_getter, span_setter

    assert my_func("ents", "ents") == ({"ents": True}, {"ents": True})
