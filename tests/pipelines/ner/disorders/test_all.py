import pytest
import spacy
from AIDS import results_aids
from alcohol import results_alcohol
from cerebrovascular_accident import results_cerebrovascular_accident
from CKD import results_ckd
from congestive_heart_failure import results_congestive_heart_failure
from connective_tissue_disease import results_connective_tissue_disease
from COPD import results_copd
from dementia import results_dementia
from diabetes import results_diabetes
from hemiplegia import results_hemiplegia
from leukemia import results_leukemia
from liver_disease import results_liver_disease
from lymphoma import results_lymphoma
from myocardial_infarction import results_myocardial_infarction
from peptic_ulcer_disease import results_peptic_ulcer_disease
from peripheral_vascular_disease import results_peripheral_vascular_disease
from solid_tumor import results_solid_tumor
from tobacco import results_tobacco

results = dict(
    aids=results_aids,
    ckd=results_ckd,
    copd=results_copd,
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
        detailled_status,
        assign=None,
    ):
        self.disorder = disorder
        self.nlp = nlp

        self.texts = texts

        self.has_match = has_match
        self.detailled_status = (
            detailled_status
            if isinstance(detailled_status, list)
            else len(texts) * [detailled_status]
        )
        self.assign = assign if assign is not None else len(texts) * [None]

        self.nlp.add_pipe(f"eds.{disorder}")

    def check(self):
        for text, has_match, detailled_status, assign in zip(
            self.texts, self.has_match, self.detailled_status, self.assign
        ):
            doc = self.nlp(text)
            ents = doc.spans[self.disorder]

            assert len(ents) == int(has_match)

            for ent in ents:
                assert ent.label_ == self.disorder

            if not ents:
                continue

            ent = ents[0]

            assert ent._.detailled_status == detailled_status

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
