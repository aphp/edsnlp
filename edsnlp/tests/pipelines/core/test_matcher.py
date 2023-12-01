import pytest
from pytest import fixture
from thinc.config import ConfigValidationError

from edsnlp.pipelines.core.matcher import GenericMatcher
from tests.conftest import text


@fixture
def nlp(blank_nlp):
    blank_nlp.add_pipe("eds.normalizer")
    return blank_nlp


@fixture
def doc(nlp):
    return nlp(text)


@fixture
def matcher_factory(nlp):
    default_config = dict(
        attr="TEXT",
        ignore_excluded=False,
    )

    def factory(terms=None, regex=None, **kwargs):

        assert terms or regex

        config = dict(**default_config)
        config.update(kwargs)

        return GenericMatcher(
            nlp=nlp,
            terms=terms,
            regex=regex,
            **config,
        )

    return factory


def test_matcher_config_typo(nlp):
    with pytest.raises(ConfigValidationError):
        nlp.add_pipe(
            "matcher",
            config={
                "terms": {"test": ["test"]},
                "term_matcher": "exoct",
            },
        )


def test_exact_matcher_spacy_factory(nlp):
    nlp.add_pipe(
        "matcher",
        config={
            "terms": {"test": ["test"]},
            "term_matcher": "exact",
        },
    )


def test_simstring_matcher_spacy_factory(nlp):
    nlp.add_pipe(
        "matcher",
        config={
            "terms": {"test": ["test"]},
            "term_matcher": "simstring",
            "term_matcher_config": {
                "measure": "dice",
            },
        },
    )


def test_terms(doc, matcher_factory):
    matcher = matcher_factory(
        terms=dict(patient="patient", anomalie="anomalie"),
        attr="NORM",
    )
    doc = matcher(doc)
    assert len(doc.ents) == 3, "There should be two entities."


def test_regex(doc, matcher_factory):
    matcher = matcher_factory(
        regex=dict(patient=r"patient", anomalie=r"anomalie"),
        attr="TEXT",
    )
    doc = matcher(doc)
    assert len(doc.ents) == 3, "There should be two entities."


def test_space(doc, matcher_factory):
    matcher = matcher_factory(
        terms=dict(holidays=r"vacances d'été"),
        attr="NORM",
        ignore_space_tokens=True,
    )
    doc = matcher(doc)
    assert len(doc.ents) == 1, "There should be one entity."
