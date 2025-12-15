import pytest
import spacy
from autonomy import results_autonomy
from cognition import results_cognition
from frailty import results_frailty
from general_status import results_general_status
from geriatric_assessment import results_ga
from incontinence import results_incontinence
from mobility import results_mobility
from nutrition import results_nutrition
from pain import results_pain
from polymed import results_polymed
from sensory import results_sensory
from social import results_social
from thymic import results_thymic

results = dict(
    autonomy=results_autonomy,
    cognition=results_cognition,
    frailty=results_frailty,
    general_status=results_general_status,
    geriatric_assessment=results_ga,
    incontinence=results_incontinence,
    mobility=results_mobility,
    nutrition=results_nutrition,
    pain=results_pain,
    polymed=results_polymed,
    sensory=results_sensory,
    social=results_social,
    thymic=results_thymic,
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
