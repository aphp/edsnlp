from typing import TYPE_CHECKING

from edsnlp.utils.lazy_module import lazify

lazify()

if TYPE_CHECKING:
    from .core.contextual_matcher.factory import create_component as contextual_matcher
    from .core.endlines.factory import create_component as endlines
    from .core.matcher.factory import create_component as matcher
    from .core.normalizer.accents.factory import create_component as accents
    from .core.normalizer.factory import create_component as normalizer
    from .core.normalizer.pollution.factory import create_component as pollution
    from .core.normalizer.quotes.factory import create_component as quotes
    from .core.normalizer.remove_lowercase.factory import (
        create_component as remove_lowercase,
    )
    from .core.normalizer.spaces.factory import create_component as spaces
    from .core.sentences.factory import create_component as sentences
    from .core.terminology.factory import create_component as terminology
    from .llm.llm_markup_extractor.factory import (
        create_component as llm_markup_extractor,
    )
    from .llm.llm_span_qualifier.factory import create_component as llm_span_qualifier
    from .misc.consultation_dates.factory import create_component as consultation_dates
    from .misc.dates.factory import create_component as dates
    from .misc.explode import Explode as explode
    from .misc.quantities.factory import create_component as quantities
    from .misc.reason.factory import create_component as reason
    from .misc.sections.factory import create_component as sections
    from .misc.split import Split as split
    from .misc.tables.factory import create_component as tables
    from .ner.adicap.factory import create_component as adicap
    from .ner.behaviors.alcohol.factory import create_component as alcohol
    from .ner.behaviors.tobacco.factory import create_component as tobacco
    from .ner.cim10.factory import create_component as cim10
    from .ner.covid.factory import create_component as covid
    from .ner.disorders.aids.factory import create_component as aids
    from .ner.disorders.cerebrovascular_accident.factory import (
        create_component as cerebrovascular_accident,
    )
    from .ner.disorders.ckd.factory import create_component as ckd
    from .ner.disorders.congestive_heart_failure.factory import (
        create_component as congestive_heart_failure,
    )
    from .ner.disorders.connective_tissue_disease.factory import (
        create_component as connective_tissue_disease,
    )
    from .ner.disorders.copd.factory import create_component as copd
    from .ner.disorders.dementia.factory import create_component as dementia
    from .ner.disorders.diabetes.factory import create_component as diabetes
    from .ner.disorders.hemiplegia.factory import create_component as hemiplegia
    from .ner.disorders.leukemia.factory import create_component as leukemia
    from .ner.disorders.liver_disease.factory import create_component as liver_disease
    from .ner.disorders.lymphoma.factory import create_component as lymphoma
    from .ner.disorders.myocardial_infarction.factory import (
        create_component as myocardial_infarction,
    )
    from .ner.disorders.peptic_ulcer_disease.factory import (
        create_component as peptic_ulcer_disease,
    )
    from .ner.disorders.peripheral_vascular_disease.factory import (
        create_component as peripheral_vascular_disease,
    )
    from .ner.disorders.solid_tumor.factory import create_component as solid_tumor
    from .ner.drugs.factory import create_component as drugs
    from .ner.frailty.cognitive_status.factory import (
        create_component as cognitive_status,
    )
    from .ner.frailty.frailty_mentions.factory import create_component as frailty
    from .ner.frailty.functional_status.factory import (
        create_component as functional_status,
    )
    from .ner.frailty.geriatric_assessment.factory import (
        create_component as geriatric_assessment,
    )
    from .ner.frailty.global_health_status.factory import (
        create_component as global_health_status,
    )
    from .ner.frailty.incontinence_status.factory import (
        create_component as incontinence_status,
    )
    from .ner.frailty.mobility_status.factory import create_component as mobility_status
    from .ner.frailty.nutritional_status.factory import (
        create_component as nutritional_status,
    )
    from .ner.frailty.pain_status.factory import create_component as pain_status
    from .ner.frailty.polypharmacy_status.factory import (
        create_component as polypharmacy_status,
    )
    from .ner.frailty.psychological_status.factory import (
        create_component as psychological_status,
    )
    from .ner.frailty.scores.adl_score.factory import create_component as adl_score
    from .ner.frailty.scores.bref_score.factory import create_component as bref_score
    from .ner.frailty.scores.chair_stand_score.factory import (
        create_component as chair_stand_score,
    )
    from .ner.frailty.scores.clinical_frailty_scale_score.factory import (
        create_component as clinical_frailty_scale_score,
    )
    from .ner.frailty.scores.ecog_performance_status_score.factory import (
        create_component as ecog_performance_status_score,
    )
    from .ner.frailty.scores.g8_score.factory import create_component as g8_score
    from .ner.frailty.scores.gait_speed_score.factory import (
        create_component as gait_speed_score,
    )
    from .ner.frailty.scores.geriatric_depression_scale_score.factory import (
        create_component as geriatric_depression_scale_score,
    )
    from .ner.frailty.scores.iadl_score.factory import create_component as iadl_score
    from .ner.frailty.scores.mini_cog_score.factory import (
        create_component as mini_cog_score,
    )
    from .ner.frailty.scores.mini_gds_score.factory import (
        create_component as mini_gds_score,
    )
    from .ner.frailty.scores.mini_mental_state_score.factory import (
        create_component as mini_mental_state_score,
    )
    from .ner.frailty.scores.moca_score.factory import create_component as moca_score
    from .ner.frailty.scores.pain_rating_score.factory import (
        create_component as pain_rating_score,
    )
    from .ner.frailty.scores.sppb_score.factory import create_component as sppb_score
    from .ner.frailty.scores.timed_up_and_go_score.factory import (
        create_component as timed_up_and_go_score,
    )
    from .ner.frailty.sensory_status.factory import create_component as sensory_status
    from .ner.scores.charlson.factory import create_component as charlson
    from .ner.scores.elston_ellis.factory import create_component as elston_ellis
    from .ner.scores.emergency.ccmu.factory import create_component as emergency_ccmu
    from .ner.scores.emergency.gemsa.factory import create_component as emergency_gemsa
    from .ner.scores.emergency.priority.factory import (
        create_component as emergency_priority,
    )
    from .ner.scores.factory import create_component as score
    from .ner.scores.sofa.factory import create_component as sofa
    from .ner.suicide_attempt.factory import create_component as suicide_attempt
    from .ner.tnm.factory import create_component as tnm
    from .ner.umls.factory import create_component as umls
    from .qualifiers.family.factory import create_component as family
    from .qualifiers.history.factory import create_component as history
    from .qualifiers.hypothesis.factory import create_component as hypothesis
    from .qualifiers.negation.factory import create_component as negation
    from .qualifiers.reported_speech.factory import create_component as reported_speech
    from .qualifiers.reported_speech.factory import create_component as rspeech
    from .trainable.biaffine_dep_parser.factory import (
        create_component as biaffine_dep_parser,
    )
    from .trainable.embeddings.span_pooler.factory import (
        create_component as span_pooler,
    )
    from .trainable.embeddings.text_cnn.factory import create_component as text_cnn
    from .trainable.embeddings.transformer.factory import (
        create_component as transformer,
    )
    from .trainable.extractive_qa.factory import create_component as extractive_qa
    from .trainable.ner_crf.factory import create_component as ner_crf
    from .trainable.span_classifier.factory import create_component as span_classifier
    from .trainable.span_linker.factory import create_component as span_linker
