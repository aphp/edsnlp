context_examples = [
    {
        'text': "Le père du patient a eu un cancer du colon.",
        'entities': [
            {
                'start': 27,
                'end': 42,
                'context': 'FAMILY',
                'context_pymedext': 'family'
            }
        ]
    },
    {
        'text': "Antécédents familiaux : diabète.",
        'entities': [
            {
                'start': 24,
                'end': 31,
                'context': 'FAMILY',
                'context_pymedext': 'family'
            }
        ]
    },
    {
        'text': "Un relevé sanguin a été effectué.",
        'entities': [
            {
                'start': 3,
                'end': 17,
                'context': 'PATIENT',
                'context_pymedext': 'patient'
            }
        ]
    }
]

hypothesis_examples = [
    {
        'text': "Plusieurs diagnostics sont envisagés.",
        'entities': [
            {
                'start': 10,
                'end': 21,
                'hypothesis': 'HYP',
                'hypothesis_pymedext': 'hypothesis'
            }
        ]
    },
    {
        'text': "Suspicion de diabète.",
        'entities': [
            {
                'start': 13,
                'end': 20,
                'hypothesis': 'HYP',
                'hypothesis_pymedext': 'hypothesis'
            }
        ]
    },
    {
        'text': "Le ligament est rompu.",
        'entities': [
            {
                'start': 16,
                'end': 21,
                'hypothesis': 'CERT',
                'hypothesis_pymedext': 'certain'
            }
        ]
    }
]

negation_examples = [
    {
        'text': "Le patient n'est pas malade.",
        'entities': [
            {
                'start': 21,
                'end': 27,
                'polarity': 'NEG',
                'polarity_pymedext': 'neg'
            }
        ]
    },
    {
        'text': "Aucun traitement.",
        'entities': [
            {
                'start': 6,
                'end': 16,
                'polarity': 'NEG',
                'polarity_pymedext': 'neg'
            }
        ]
    },
    {
        'text': "Le scan révèle une grosseur.",
        'entities': [
            {
                'start': 8,
                'end': 14,
                'polarity': 'AFF',
                'polarity_pymedext': 'aff'
            }
        ]
    }

]
