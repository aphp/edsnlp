import pytest
import spacy
from adl import results_adl
from g8 import results_g8
from gds import results_gds
from iadl import results_iadl
from mini_gds import results_mini_gds
from mms import results_mms
from moca import results_moca
from ps import results_ps
from rockwood import results_rockwood
from tug import results_tug

from tests.pipelines.ner.frailty.gait_speed import results_gait_speed

results = dict(
    adl={"results": results_adl, "domain": "autonomy"},
    iadl={"results": results_iadl, "domain": "autonomy"},
    mms={"results": results_mms, "domain": "cognition"},
    moca={"results": results_moca, "domain": "cognition"},
    tug={"results": results_tug, "domain": "mobility"},
    gait_speed={"results": results_gait_speed, "domain": "mobility"},
    gds={"results": results_gds, "domain": "thymic"},
    mini_gds={"results": results_mini_gds, "domain": "thymic"},
    g8={"results": results_g8, "domain": "g8"},
    ps={"results": results_ps, "domain": "general_status"},
    rockwood={"results": results_rockwood, "domain": "general_status"},
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
