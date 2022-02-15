from pytest import fixture

from edsnlp.pipelines.core.matcher import GenericMatcher


@fixture
def matcher_factory(blank_nlp):

    default_config = dict(
        attr="TEXT",
        filter_matches=True,
        on_ents_only=False,
        ignore_excluded=False,
    )

    def factory(terms=None, regex=None, **kwargs):

        assert terms or regex

        config = dict(**default_config)
        config.update(kwargs)

        return GenericMatcher(
            nlp=blank_nlp,
            terms=terms,
            regex=regex,
            **config,
        )

    return factory


def test_terms(blank_doc, matcher_factory):
    matcher = matcher_factory(
        terms=dict(patient="patient", anomalie="anomalie"),
        attr="NORM",
    )
    doc = matcher(blank_doc)
    assert len(doc.ents) == 3, "There should be two entities."


def test_regex(blank_doc, matcher_factory):
    matcher = matcher_factory(
        regex=dict(patient=r"patient", anomalie=r"anomalie"),
        attr="TEXT",
    )
    doc = matcher(blank_doc)
    assert len(doc.ents) == 3, "There should be two entities."
