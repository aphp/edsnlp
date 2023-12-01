from edsnlp.pipelines.misc.measurements.patterns import common_measurements, number_terms, value_range_terms, units_config, unit_divisors, stopwords_unitless, stopwords_measure_unit


#######################################
# ## CONFIG TO PRETREAT THE BRAT DIR ###
# ######################################

# Regex which identifies in group 1 the beginning of a span and in group 2
# the end of the same span
regex_convert_spans = r"^(\d+).*\s(\d+)$"

# Label of the entities containing the measurement and possibly
# other random entities
label_key = "BIO_comp"

# Labels of the entities It is possible not to consider during matching
labels_to_remove = ["BIO"]

# Labels of the entities which can be linked to a measurement
labels_linkable_to_measurement = ["BIO"]


##########################################################
# ## PIPE TO MATCH MEASUREMENTS IN `label_key` ENTITIES ###
# #########################################################

# Config of the normalizer pipe used in entities labeled `label_key`
config_normalizer_from_label_key = dict(
    lowercase=True,
    accents=True,
    quotes=True,
    pollution=True,
)

# Terms which will make the measurements pipe match a positive measurement
positive_terms_from_label_key = ("positifs", "positives", "positivites")
# We create a list to match abbreviations of the positive words. This list will
# be the final dictionnary used to match the positive measurements.
positive_terms_from_label_key = [
    word[: i + 1]
    for word in positive_terms_from_label_key
    for i in range(min(len(word) - 1, 1), len(word))
]
# Symbols which will make the measurements pipe match a positive measurement
positive_symbols_from_label_key = ("\+", "p")
# To match symbols, we create regex
positive_regex_from_label_key = [
    r"^[^a-zA-Z0-9]*(?:% s)" % "|".join(positive_symbols_from_label_key)
    + r"[^a-zA-Z0-9]*$"
]

# Terms which will make the measurements pipe match a negative measurement
negative_terms_from_label_key = (
    "negatifs",
    "negatives",
    "negativites",
    "absences",
    "absents",
)
# We create a list to match abbreviations of the negative words. This list will
# be the final dictionnary used to match the negative measurements.
negative_terms_from_label_key = [
    word[: i + 1]
    for word in negative_terms_from_label_key
    for i in range(min(len(word) - 1, 1), len(word))
]
# Symbols which will make the measurements pipe match a positive measurement
negative_symbols_from_label_key = ("\-", "n")
# To match symbols, we create regex
negative_regex_from_label_key = [
    r"^[^a-zA-Z0-9]*(?:% s)" % "|".join(negative_symbols_from_label_key)
    + r"[^a-zA-Z0-9]*$"
]

# Terms which will make the measurements pipe match a normal measurement
normal_terms_from_label_key = ("normales", "normaux", "normalisations", "normalites")
# We create a list to match abbreviations of the normal words. This list will
# be the final dictionnary used to match the normal measurements.
normal_terms_from_label_key = [
    word[: i + 1]
    for word in normal_terms_from_label_key
    for i in range(min(len(word) - 1, 1), len(word))
]

# Custom mesurements mainly to include custom positive, negative
# and normal measurements
measurements_from_label_key = {
    "eds.weight": {
        "unit": "kg",
        "unitless_patterns": [
            {
                "terms": ["poids", "poid", "pese", "pesant", "pesait", "pesent"],
                "ranges": [
                    {"min": 0, "max": 200, "unit": "kg"},
                    {"min": 200, "unit": "g"},
                ],
            }
        ],
    },
    "eds.size": {
        "unit": "m",
        "unitless_patterns": [
            {
                "terms": [
                    "mesure",
                    "taille",
                    "mesurant",
                    "mesurent",
                    "mesurait",
                    "mesuree",
                    "hauteur",
                    "largeur",
                    "longueur",
                ],
                "ranges": [
                    {"min": 0, "max": 3, "unit": "m"},
                    {"min": 3, "unit": "cm"},
                ],
            }
        ],
    },
    "eds.bmi": {
        "unit": "kg_per_m2",
        "unitless_patterns": [
            {"terms": ["imc", "bmi"], "ranges": [{"unit": "kg_per_m2"}]}
        ],
    },
    "eds.volume": {"unit": "m3", "unitless_patterns": []},
    "eds.bool": {
        "unit": "bool",
        "valueless_patterns": [
            {
                "terms": positive_terms_from_label_key,
                "regex": positive_regex_from_label_key,
                "measurement": {
                    "value_range": "=",
                    "value": 1,
                    "unit": "bool",
                },
            },
            {
                "terms": negative_terms_from_label_key,
                "regex": negative_regex_from_label_key,
                "measurement": {
                    "value_range": "=",
                    "value": 0,
                    "unit": "bool",
                },
            },
            {
                "terms": normal_terms_from_label_key,
                "measurement": {
                    "value_range": "=",
                    "value": 0.5,
                    "unit": "bool",
                },
            },
        ],
    },
}

# Config of the measurement pipe used in entities labeled `label_key`
config_measurements_from_label_key = dict(
    measurements=measurements_from_label_key,
    units_config=units_config,
    number_terms=number_terms,
    value_range_terms=value_range_terms,
    unit_divisors=unit_divisors,
    stopwords_unitless=stopwords_unitless,
    stopwords_measure_unit=stopwords_measure_unit,
    measure_before_unit=False,
    ignore_excluded=True,
    attr="NORM",
    all_measurements=True,
    parse_tables=False,
    parse_doc=True,
)


############################################
# ## PIPE TO MATCH MEASUREMENTS IN TABLES ###
# ###########################################

# Config of the normalizer pipe used in tables
config_normalizer_from_tables = dict(
    lowercase=True,
    accents=True,
    quotes=True,
    pollution=True,
)

# Terms which will make the measurements pipe match a positive measurement
positive_terms_from_tables = ("positifs", "positives", "positivites")
# We create a list to match abbreviations of the positive words. This list will
# be the final dictionnary used to match the positive measurements.
positive_terms_from_tables = [
    word[: i + 1]
    for word in positive_terms_from_tables
    for i in range(min(len(word) - 1, 1), len(word))
]
# Symbols which will make the measurements pipe match a positive measurement
positive_symbols_from_tables = ("\+", "p")
# To match symbols, we create regex
positive_regex_from_tables = [
    r"^[^a-zA-Z0-9]*(?:% s)" % "|".join(positive_symbols_from_tables) + r"[^a-zA-Z0-9]*$"
]

# Terms which will make the measurements pipe match a negative measurement
negative_terms_from_tables = (
    "negatifs",
    "negatives",
    "negativites",
    "absences",
    "absents",
)
# We create a list to match abbreviations of the negative words. This list will
# be the final dictionnary used to match the negative measurements.
negative_terms_from_tables = [
    word[: i + 1]
    for word in negative_terms_from_tables
    for i in range(min(len(word) - 1, 1), len(word))
]
# Symbols which will make the measurements pipe match a positive measurement
negative_symbols_from_tables = ("\-", "n")
# To match symbols, we create regex
negative_regex_from_tables = [
    r"^[^a-zA-Z0-9]*(?:% s)" % "|".join(negative_symbols_from_tables) + r"[^a-zA-Z0-9]*$"
]

# Terms which will make the measurements pipe match a normal measurement
normal_terms_from_tables = ("normales", "normaux", "normalisations", "normalites")
# We create a list to match abbreviations of the normal words. This list will
# be the final dictionnary used to match the normal measurements.
normal_terms_from_tables = [
    word[: i + 1]
    for word in normal_terms_from_tables
    for i in range(min(len(word) - 1, 1), len(word))
]

# Custom mesurements mainly to include custom positive, negative
# and normal measurements
measurements_from_tables = {
    "eds.weight": {
        "unit": "kg",
        "unitless_patterns": [
            {
                "terms": ["poids", "poid", "pese", "pesant", "pesait", "pesent"],
                "ranges": [
                    {"min": 0, "max": 200, "unit": "kg"},
                    {"min": 200, "unit": "g"},
                ],
            }
        ],
    },
    "eds.size": {
        "unit": "m",
        "unitless_patterns": [
            {
                "terms": [
                    "mesure",
                    "taille",
                    "mesurant",
                    "mesurent",
                    "mesurait",
                    "mesuree",
                    "hauteur",
                    "largeur",
                    "longueur",
                ],
                "ranges": [
                    {"min": 0, "max": 3, "unit": "m"},
                    {"min": 3, "unit": "cm"},
                ],
            }
        ],
    },
    "eds.bmi": {
        "unit": "kg_per_m2",
        "unitless_patterns": [
            {"terms": ["imc", "bmi"], "ranges": [{"unit": "kg_per_m2"}]}
        ],
    },
    "eds.volume": {"unit": "m3", "unitless_patterns": []},
    "eds.bool": {
        "unit": "bool",
        "valueless_patterns": [
            {
                "terms": positive_terms_from_tables,
                "regex": positive_regex_from_tables,
                "measurement": {
                    "value_range": "=",
                    "value": 1,
                    "unit": "bool",
                },
            },
            {
                "terms": negative_terms_from_tables,
                "regex": negative_regex_from_tables,
                "measurement": {
                    "value_range": "=",
                    "value": 0,
                    "unit": "bool",
                },
            },
            {
                "terms": normal_terms_from_tables,
                "measurement": {
                    "value_range": "=",
                    "value": 0.5,
                    "unit": "bool",
                },
            },
        ],
    },
}

# Config of the measurement pipe used in tables
config_measurements_from_tables = dict(
    measurements=measurements_from_tables,
    units_config=units_config,
    number_terms=number_terms,
    value_range_terms=value_range_terms,
    unit_divisors=unit_divisors,
    stopwords_unitless=stopwords_unitless,
    stopwords_measure_unit=stopwords_measure_unit,
    measure_before_unit=False,
    ignore_excluded=True,
    attr="NORM",
    all_measurements=True,
    parse_tables=True,
    parse_doc=False,
)
