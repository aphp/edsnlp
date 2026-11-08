import pytest
import spacy
from adl import results_adl
from bref import results_bref
from chair_stand import results_chair_stand
from en_eva import results_en_eva
from g8 import results_g8
from gait_speed import results_gait_speed
from gds import results_gds
from iadl import results_iadl
from mini_cog import results_mini_cog
from mini_gds import results_mini_gds
from mms import results_mms
from moca import results_moca
from ps import results_ps
from rockwood import results_clinical_frailty_scale
from sppb import results_sppb
from tug import results_tug

from edsnlp.pipes.ner.frailty.scores.base import FrailtyScoreMatcher
from edsnlp.pipes.ner.frailty.scores.utils import (
    float_regex,
    make_find_value_and_reference,
)

results = dict(
    adl_score={"results": results_adl, "domain": "functional_status"},
    iadl_score={"results": results_iadl, "domain": "functional_status"},
    mini_mental_state_score={"results": results_mms, "domain": "cognitive_status"},
    moca_score={"results": results_moca, "domain": "cognitive_status"},
    timed_up_and_go_score={"results": results_tug, "domain": "mobility_status"},
    gait_speed_score={"results": results_gait_speed, "domain": "mobility_status"},
    geriatric_depression_scale_score={
        "results": results_gds,
        "domain": "psychological_status",
    },
    mini_gds_score={"results": results_mini_gds, "domain": "psychological_status"},
    g8_score={"results": results_g8, "domain": "g8"},
    ecog_performance_status_score={
        "results": results_ps,
        "domain": "global_health_status",
    },
    clinical_frailty_scale_score={
        "results": results_clinical_frailty_scale,
        "domain": "global_health_status",
    },
    bref_score={"results": results_bref, "domain": "cognitive_status"},
    chair_stand_score={"results": results_chair_stand, "domain": "mobility_status"},
    pain_rating_score={"results": results_en_eva, "domain": "pain_status"},
    mini_cog_score={"results": results_mini_cog, "domain": "cognitive_status"},
    sppb_score={"results": results_sppb, "domain": "mobility_status"},
)


@pytest.fixture
def normalized_nlp(lang):
    model = spacy.blank(lang)
    model.add_pipe("eds.sentences")
    model.add_pipe("eds.normalizer")
    return model


class FrailtyScoreTester:
    def __init__(
        self,
        score,
        domain,
        nlp,
        results,
    ):
        self.score = score
        self.domain = domain
        self.nlp = nlp
        self.results = results
        self.nlp.add_pipe(f"eds.{score}")

    def check(self):
        for input, expected in self.results:
            pred = self.nlp(input)
            if expected is None:
                # No match expected
                assert len(pred.ents) == 0
                continue
            assert self.domain in pred.spans
            assert self.score in pred.spans
            value, severity = expected
            assert len(pred.ents) == 1
            ent = pred.ents[0]
            assert ent.has_extension(self.score)
            assert ent._.get("value") == value
            assert ent._.get(self.domain) == severity


@pytest.mark.parametrize(
    "score",
    list(results.keys()),
)
def test_frailty_scores(normalized_nlp, score):
    score_results = results[score]["results"]
    domain = results[score]["domain"]
    tester = FrailtyScoreTester(
        score,
        domain,
        normalized_nlp,
        score_results,
    )
    tester.check()


# Various tests on base class for edge cases

default_score_normalization = make_find_value_and_reference([10], 10)

test_base_class = dict(
    span_getter_none=dict(
        patterns=dict(
            source="test_score",
            regex=[r"test score"],
            assign=[
                dict(
                    name="value",
                    regex=rf"({float_regex})",
                    window=(0, 7),
                    reduce_mode="keep_last",
                ),
            ],
        ),
        span_setter=None,
        score_normalization=default_score_normalization,
        examples=[
            {"text": "test score 7", "results": ["test score 7"]},
        ],
    ),
    assigned_required=dict(
        patterns=dict(
            source="test_score",
            regex=[r"test score"],
            assign=[
                dict(
                    name="value",
                    regex=rf"({float_regex})",
                    window=(0, 7),
                    reduce_mode="keep_last",
                    required=True,
                ),
            ],
        ),
        span_setter={"ents": True, "test": True},
        score_normalization=default_score_normalization,
        examples=[
            {"text": "test score", "results": []},
        ],
    ),
    mutliple_limits=dict(
        patterns=dict(
            source="test_score",
            regex=[r"test score"],
            assign=[
                dict(
                    name="value",
                    regex=rf"({float_regex})",
                    window=(0, 35),
                    reduce_mode="keep_last",
                ),
                dict(
                    name="limit_1",
                    regex="(limit)",
                    window=(0, 35),
                ),
                dict(name="limit_2", regex="(border)", window=(0, 35)),
            ],
        ),
        span_setter={"ents": True, "test": True},
        score_normalization=default_score_normalization,
        examples=[
            {"text": "test score 5 7/10", "results": ["test score 5 7/10"]},
            {"text": "test score 5 border limit 7/10", "results": ["test score 5"]},
        ],
    ),
)


@pytest.mark.parametrize(
    "test_case",
    list(test_base_class.keys()),
)
def test_frailty_score_base(normalized_nlp, test_case):
    test_params = test_base_class[test_case]
    patterns = test_params["patterns"]
    span_setter = test_params["span_setter"]
    score_normalization = test_params["score_normalization"]
    examples = test_params["examples"]
    test_score = FrailtyScoreMatcher(
        normalized_nlp,
        name="test_score",
        domain="test",
        patterns=patterns,
        span_setter=span_setter,
        score_normalization=score_normalization,
        severity_assigner=lambda x: "other",
        label="test",
        include_assigned=True,
    )
    for example in examples:
        doc = normalized_nlp(example["text"])
        doc = test_score(doc)
        assert len(doc.ents) == len(example["results"])
        for ent, result in zip(doc.ents, example["results"]):
            assert ent.text == result
