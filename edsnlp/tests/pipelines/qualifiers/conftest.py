from pytest import fixture


@fixture(params=[True, False])
def blank_nlp(blank_nlp, request, lang):
    if request.param:
        blank_nlp.add_pipe("normalizer")
    return blank_nlp
