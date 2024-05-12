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
    from .misc.consultation_dates.factory import create_component as consultation_dates
    from .misc.dates.factory import create_component as dates
    from .misc.quantities.factory import create_component as quantities
    from .misc.reason.factory import create_component as reason
    from .misc.sections.factory import create_component as sections
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
    from .ner.scores.charlson.factory import create_component as charlson
    from .ner.scores.elston_ellis.factory import create_component as elston_ellis
    from .ner.scores.emergency.ccmu.factory import create_component as emergency_ccmu
    from .ner.scores.emergency.gemsa.factory import create_component as emergency_gemsa
    from .ner.scores.emergency.priority.factory import create_component as emergency_priority
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
    from .trainable.ner_crf.factory import create_component as ner_crf
    from .trainable.biaffine_dep_parser.factory import create_component as biaffine_dep_parser
    from .trainable.extractive_qa.factory import create_component as extractive_qa
    from .trainable.span_classifier.factory import create_component as span_classifier
    from .trainable.span_linker.factory import create_component as span_linker
    from .trainable.embeddings.span_pooler.factory import create_component as span_pooler
    from .trainable.embeddings.transformer.factory import create_component as transformer
    from .trainable.embeddings.text_cnn.factory import create_component as text_cnn
    from .misc.split import Split as split
