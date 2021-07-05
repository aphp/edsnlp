"""
These section titles were extracted from a work performed by Ivan Lerner at AP-HP.
It supplied a number of documents annotated for section titles.

The section titles were reviewed by Gilles Chatellier, who gave meaningful insights.

See sections/section-dataset notebook for detail.
"""

allergies = ['allergies', 'allergies :']

antecedents = ['antecedents', 'antecedents :', 'antecedents medicaux et chirurgicaux', 'antecedents personnels']

antecedents_familiaux = ['antecedents familiaux']

traitements_entree = ['attitude therapeutique initiale', "traitement a l'entree", 'traitement actuel :',
                      'traitement en cours', "traitements a l'entree"]

conclusion = ['au total', 'conclusion', 'conclusion :', 'conclusion de sortie', 'syntese medicale / conclusion',
              'synthese', 'synthese medicale', 'synthese medicale :', 'synthese medicale/conclusion']

conclusion_entree = ["conclusion a l'entree"]

habitus = ['contexte familial et social', 'habitus', 'mode de vie', 'mode de vie - scolarite', 'mode de vie :',
           'situation sociale, mode de vie']

correspondants = ['correspondants']

diagnostic = ['diagnostic retenu']

donnees_biometriques_entree = ["donnees biometriques et parametres vitaux a l'entree",
                               "parametres vitaux et donnees biometriques a l'entree"]

examens = ['examen clinique', 'examen clinique :', "examen clinique a l'entree", "examen clinique a l'entree :"]

examens_complementaires = ['examen(s) complementaire(s) :', 'examens complementaires', 'examens complementaires :',
                           "examens complementaires a l'entree", 'examens complementaires realises pendant le sejour',
                           'examens para-cliniques']

facteurs_de_risques = ['facteurs de risque', 'facteurs de risques']

histoire_de_la_maladie = ['histoire de la maladie', 'histoire de la maladie - explorations', 'histoire de la maladie :',
                          'histoire de la maladie actuelle', 'histoire du poids :', 'histoire recente',
                          'histoire recente de la maladie', 'rappel clinique', 'resume', 'resume clinique']

actes = ['intervention']

motif = ['motif', "motif d'hospitalisation", "motif de l'hospitalisation", "motif de l'hospitalisation :"]

prescriptions = ['prescriptions de sortie :', 'prescriptions medicales de sortie',
                 'prescriptions medicales de sortie :']

traitements_sortie = ['traitement de sortie', 'traitement de sortie :']

sections = {
    'allergies': allergies,
    'antécédents': antecedents,
    'antécédents familiaux': antecedents_familiaux,
    'traitements entrée': traitements_entree,
    'conclusion': conclusion,
    'conclusion entrée': conclusion_entree,
    'habitus': habitus,
    'correspondants': correspondants,
    'diagnostic': diagnostic,
    'données biométriques entrée': donnees_biometriques_entree,
    'examens': examens,
    'examens complémentaires': examens_complementaires,
    'facteurs de risques': facteurs_de_risques,
    'histoire de la maladie': histoire_de_la_maladie,
    'actes': actes,
    'motif': motif,
    'prescriptions': prescriptions,
    'traitements sortie': traitements_sortie,
}
