import pytest
import spacy
from test_AIDS import results_aids
from test_alcohol import results_alcohol
from test_cerebrovascular_accident import results_cerebrovascular_accident
from test_CKD import results_ckd
from test_congestive_heart_failure import results_congestive_heart_failure
from test_connective_tissue_disease import results_connective_tissue_disease
from test_COPD import results_copd
from test_dementia import results_dementia
from test_diabetes import results_diabetes
from test_hemiplegia import results_hemiplegia
from test_leukemia import results_leukemia
from test_liver_disease import results_liver_disease
from test_lymphoma import results_lymphoma
from test_myocardial_infarction import results_myocardial_infarction
from test_peptic_ulcer_disease import results_peptic_ulcer_disease
from test_peripheral_vascular_disease import results_peripheral_vascular_disease
from test_solid_tumor import results_solid_tumor
from test_tobacco import results_tobacco

results = dict(
    AIDS=results_aids,
    CKD=results_ckd,
    COPD=results_copd,
    alcohol=results_alcohol,
    cerebrovascular_accident=results_cerebrovascular_accident,
    congestive_heart_failure=results_congestive_heart_failure,
    connective_tissue_disease=results_connective_tissue_disease,
    dementia=results_dementia,
    diabetes=results_diabetes,
    hemiplegia=results_hemiplegia,
    leukemia=results_leukemia,
    liver_disease=results_liver_disease,
    lymphoma=results_lymphoma,
    myocardial_infarction=results_myocardial_infarction,
    peptic_ulcer_disease=results_peptic_ulcer_disease,
    peripheral_vascular_disease=results_peripheral_vascular_disease,
    solid_tumor=results_solid_tumor,
    tobacco=results_tobacco,
)


@pytest.fixture
def normalized_nlp(lang):
    model = spacy.blank(lang)
    model.add_pipe("eds.sentences")
    model.add_pipe("eds.normalizer")
    return model


class DisorderTester:
    def __init__(
        self,
        disorder,
        nlp,
        texts,
        has_match,
        status_,
        assign=None,
    ):
        self.disorder = disorder
        self.nlp = nlp

        self.texts = texts

        self.has_match = has_match
        self.status_ = status_ if isinstance(status_, list) else len(texts) * [status_]
        self.assign = assign if assign is not None else len(texts) * [None]

        self.nlp.add_pipe(f"eds.{disorder}")

    def check(self):
        for text, has_match, status_, assign in zip(
            self.texts, self.has_match, self.status_, self.assign
        ):
            doc = self.nlp(text)
            ents = doc.spans[self.disorder]

            assert len(ents) == int(has_match)

            if not ents:
                continue

            ent = ents[0]

            assert ent._.status_ == status_

            if assign is not None:
                for key, value in assign.items():
                    assert repr(value) == repr(assign[key])


@pytest.mark.parametrize(
    "disorder",
    list(results.keys()),
)
def test_disorder(normalized_nlp, disorder):

    result = results[disorder]

    expect = DisorderTester(
        disorder,
        normalized_nlp,
        **result,
    )

    expect.check()
