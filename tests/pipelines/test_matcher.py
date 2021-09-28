from edsnlp.pipelines.matcher import GenericMatcher
from pytest import fixture, mark


@fixture
def matcher_factory(blank_nlp):

    default_config = dict(
        attr="TEXT",
        fuzzy=False,
        fuzzy_kwargs=None,
        filter_matches=True,
        on_ents_only=False,
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


@mark.parametrize("fuzzy", [True, False])
def test_terms(doc, matcher_factory, fuzzy):
    matcher = matcher_factory(
        terms=dict(patient="patient", anomalie="anomalie"),
        attr="NORM",
        fuzzy=fuzzy,
    )
    doc = matcher(doc)
    assert len(doc.ents) == 2, "There should be two entities."
