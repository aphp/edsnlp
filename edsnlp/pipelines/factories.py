# flake8: noqa: F811
from .core.context.factory import create_component as context
from .core.contextual_matcher.factory import create_component as contextual_matcher
from .core.endlines.factory import create_component as endlines
from .core.matcher.factory import create_component as matcher
from .core.normalizer.accents.factory import create_component as accents
from .core.normalizer.factory import create_component as normalizer
from .core.normalizer.lowercase.factory import remove_lowercase
from .core.normalizer.pollution.factory import create_component as pollution
from .core.normalizer.quotes.factory import create_component as quotes
from .core.sentences.factory import create_component as sentences
from .core.terminology.factory import create_component as terminology
from .misc.consultation_dates.factory import create_component as consultation_dates
from .misc.dates.factory import create_component as dates
from .misc.measurements.factory import create_component as measurements
from .misc.reason.factory import create_component as reason
from .misc.sections.factory import create_component as sections
from .ner.adicap.factory import create_component as adicap
from .ner.cim10.factory import create_component as cim10
from .ner.comorbidities.adrenal_insufficiency.factory import (
    create_component as adrenal_insufficiency,
)
from .ner.comorbidities.AIDS.factory import create_component as AIDS
from .ner.comorbidities.alcohol.factory import create_component as alcohol
from .ner.comorbidities.atrial_fibrillation.factory import (
    create_component as atrial_fibrillation,
)
from .ner.comorbidities.cerebrovascular_accident.factory import (
    create_component as cerebrovascular_accident,
)
from .ner.comorbidities.CKD.factory import create_component as CKD
from .ner.comorbidities.congestive_heart_failure.factory import (
    create_component as congestive_heart_failure,
)
from .ner.comorbidities.connective_tissue_disease.factory import (
    create_component as connective_tissue_disease,
)
from .ner.comorbidities.COPD.factory import create_component as COPD
from .ner.comorbidities.dementia.factory import create_component as dementia
from .ner.comorbidities.diabetes.factory import create_component as diabetes
from .ner.comorbidities.hemiplegia.factory import create_component as hemiplegia
from .ner.comorbidities.hypertension.factory import create_component as hypertension
from .ner.comorbidities.leukemia.factory import create_component as leukemia
from .ner.comorbidities.liver_disease.factory import create_component as liver_disease
from .ner.comorbidities.lymphoma.factory import create_component as lymphoma
from .ner.comorbidities.myasthenia.factory import create_component as myasthenia
from .ner.comorbidities.myocardial_infarction.factory import (
    create_component as myocardial_infarction,
)
from .ner.comorbidities.peptic_ulcer_disease.factory import (
    create_component as peptic_ulcer_disease,
)
from .ner.comorbidities.peripheral_vascular_disease.factory import (
    create_component as peripheral_vascular_disease,
)
from .ner.comorbidities.solid_tumor.factory import create_component as solid_tumor
from .ner.comorbidities.tobacco.factory import create_component as tobacco
from .ner.covid.factory import create_component as covid
from .ner.drugs.factory import create_component as drugs
from .ner.scores.charlson.factory import create_component as charlson
from .ner.scores.emergency.ccmu.factory import create_component as ccmu
from .ner.scores.emergency.gemsa.factory import create_component as gemsa
from .ner.scores.emergency.priority.factory import create_component as priority
from .ner.scores.factory import create_component as score
from .ner.scores.sofa.factory import create_component as sofa
from .ner.scores.tnm.factory import create_component as tnm
from .qualifiers.family.factory import create_component as family
from .qualifiers.history.factory import create_component as history
from .qualifiers.hypothesis.factory import create_component as hypothesis
from .qualifiers.negation.factory import create_component as negation
from .qualifiers.reported_speech.factory import create_component as rspeech
from .trainable.nested_ner import create_component as nested_ner
