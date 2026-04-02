import pytest
import spacy
from cognition import results_cognition
from frailty import results_frailty
from functional_status import results_functional_status
from general_status import results_general_status
from geriatric_assessment import results_ga
from incontinence import results_incontinence
from mobility import results_mobility
from nutrition import results_nutrition
from pain import results_pain
from polymed import results_polymed
from psychological_status import results_psychological_status
from sensory import results_sensory
from social import results_social

results = dict(
    functional_status=results_functional_status,
    cognitive_status=results_cognition,
    frailty_mentions=results_frailty,
    global_health_status=results_general_status,
    geriatric_assessment=results_ga,
    incontinence_status=results_incontinence,
    mobility_status=results_mobility,
    nutritional_status=results_nutrition,
    pain_status=results_pain,
    polypharmacy_status=results_polymed,
    sensory_status=results_sensory,
    social_status=results_social,
    psychological_status=results_psychological_status,
)


@pytest.fixture
def normalized_nlp(lang):
    model = spacy.blank(lang)
    model.add_pipe("eds.sentences")
    model.add_pipe("eds.normalizer")
    return model


class FrailtyDomainTester:
    def __init__(
        self,
        domain,
        nlp,
        results,
    ):
        self.domain = domain
        self.nlp = nlp
        self.results = results
        self.nlp.add_pipe(f"eds.{domain}")

    def check(self):
        for input, expected in self.results:
            pred = self.nlp(input)
            if expected is None:
                # No match expected
                assert len(pred.ents) == 0
                continue
            assert len(pred.ents) == 1
            ent = pred.ents[0]
            assert ent.has_extension(self.domain)
            assert ent._.get(self.domain) == expected


@pytest.mark.parametrize(
    "domain",
    list(results.keys()),
)
def test_frailty(normalized_nlp, domain):
    domain_results = results[domain]
    tester = FrailtyDomainTester(
        domain,
        normalized_nlp,
        domain_results,
    )
    tester.check()
