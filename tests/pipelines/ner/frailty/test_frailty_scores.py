import pytest
import spacy
from adl import results_adl

results = dict(adl={"results": results_adl, "domain": "autonomy"})


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
            assert ent._.get(self.score) == value
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
