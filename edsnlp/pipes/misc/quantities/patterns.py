# ruff:noqa:E501

number_terms = {
    "0.125": ["⅛"],
    "0.16666666": ["⅙"],
    "0.2": ["⅕"],
    "0.25": ["¼"],
    "0.3333333": ["⅓"],
    "0.5": ["½"],
    "2.5": ["21/2"],
    "1": ["un", "une"],
    "2": ["deux"],
    "3": ["trois"],
    "4": ["quatre"],
    "5": ["cinq"],
    "6": ["six"],
    "7": ["sept"],
    "8": ["huit"],
    "9": ["neuf"],
    "10": ["dix"],
    "11": ["onze"],
    "12": ["douze"],
    "13": ["treize"],
    "14": ["quatorze"],
    "15": ["quinze"],
    "16": ["seize"],
    "17": ["dix-sept", "dix sept"],
    "18": ["dix-huit", "dix huit"],
    "19": ["dix-neuf", "dix neuf"],
    "20": ["vingt", "vingts"],
    "30": ["trente"],
    "40": ["quarante"],
    "50": ["cinquante"],
    "60": ["soixante"],
    "70": ["soixante dix", "soixante-dix"],
    "80": ["quatre vingt", "quatre-vingt", "quatre vingts", "quatre-vingts"],
    "90": ["quatre vingt dix", "quatre-vingt-dix"],
    "100": ["cent"],
    "500": ["cinq cent", "cinq-cent"],
    "1000": ["mille", "milles"],
}


number_regex = r"""(?x)
# no digit or floating point number prefix before
(?<![0-9][.,]?)
# integer part like 123 or 1 234
(?:
    0
    |[1-9][0-9]*(?:\ \d{3})*
)
(?:
    # floating point surounded by spaces
    \ +[,.]\ +\d+
    # floating point w/o space
    | [,.]\d+
    # fractions or slash groups
    | (?:(?:\ /\ |/)[1-9][0-9]*(?:\ \d{3})*)+
)?"""


common_quantities = {
    "weight": {
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
    "size": {
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
    "bmi": {
        "unit": "kg_per_m2",
        "unitless_patterns": [
            {"terms": ["imc", "bmi"], "ranges": [{"unit": "kg_per_m2"}]}
        ],
    },
    "volume": {"unit": "m3", "unitless_patterns": []},
}

unit_divisors = ["/", "par"]

stopwords = ["par", "sur", "de", "a", ",", "et", "-", "à"]

# Should we only make accented patterns and expect the user to use
# `eds.normalizer` component first ?
range_patterns = [
    ("De", "à"),
    ("De", "a"),
    ("de", "à"),
    ("de", "a"),
    ("Entre", "et"),
    ("entre", "et"),
    (None, "a"),
    (None, "à"),
    (None, "-"),
]

# fmt: off
units_config = {
    # Mass
    "kg":  {"dim": "mass", "degree": 1, "scale": 1e3, "terms": ["kgramme", "kilogramme", "kilo-gramme", "kgrammes", "kilogrammes", "kilo-grammes", "kgr", "kilogr", "kilo-gr", "kg", "kilog", "kilo-g"], "followed_by": "g"},
    "hg":  {"dim": "mass", "degree": 1, "scale": 1e2, "terms": ["hgramme", "hectogramme", "hecto-gramme", "hgrammes", "hectogrammes", "hecto-grammes", "hgr", "hectogr", "hecto-gr", "hg", "hectog", "hecto-g"], "followed_by": None},
    "dag": {"dim": "mass", "degree": 1, "scale": 1e1, "terms": ["dagramme", "decagramme", "deca-gramme", "dagrammes", "decagrammes", "deca-grammes", "dagr", "decagr", "deca-gr", "dag", "decag", "deca-g"], "followed_by": None},
    "g":   {"dim": "mass", "degree": 1, "scale": 1e0, "terms": ["gramme", "grammes", "gr", "g"], "followed_by": None},
    "dg":  {"dim": "mass", "degree": 1, "scale": 1e-1, "terms": ["dgramme", "decigramme", "deci-gramme", "dgrammes", "decigrammes", "deci-grammes", "dgr", "decigr", "deci-gr", "dg", "decig", "deci-g"], "followed_by": None},
    "cg":  {"dim": "mass", "degree": 1, "scale": 1e-2, "terms": ["cgramme", "centigramme", "centi-gramme", "cgrammes", "centigrammes", "centi-grammes", "cgr", "centigr", "centi-gr", "cg", "centig", "centi-g"], "followed_by": None},
    "mg":  {"dim": "mass", "degree": 1, "scale": 1e-3, "terms": ["mgramme", "milligramme", "milli-gramme", "miligramme", "mili-gramme", "mgrammes", "milligrammes", "milli-grammes", "miligrammes", "mili-grammes", "mgr", "milligr", "milli-gr", "miligr", "mili-gr", "mg", "millig", "milli-g", "milig", "mili-g"], "followed_by": None},
    "μg":  {"dim": "mass", "degree": 1, "scale": 1e-6, "terms": ["μgramme", "ugramme", "µgramme", "microgramme", "micro-gramme", "μgrammes", "ugrammes", "µgrammes", "microgrammes", "micro-grammes", "μgr", "ugr", "µgr", "microgr", "micro-gr", "μg", "ug", "µg", "microg", "micro-g"], "followed_by": None},
    "ng":  {"dim": "mass", "degree": 1, "scale": 1e-9, "terms": ["ngramme", "nanogramme", "nano-gramme", "ngrammes", "nanogrammes", "nano-grammes", "ngr", "nanogr", "nano-gr", "ng", "nanog", "nano-g"], "followed_by": None},
    "pg":  {"dim": "mass", "degree": 1, "scale": 1e-12, "terms": ["pgramme", "picogramme", "pico-gramme", "pgrammes", "picogrammes", "pico-grammes", "pgr", "picogr", "pico-gr", "pg", "picog", "pico-g"], "followed_by": None},
    "fg":  {"dim": "mass", "degree": 1, "scale": 1e-15, "terms": ["fgramme", "femtogramme", "femto-gramme", "fgrammes", "femtogrammes", "femto-grammes", "fgr", "femtogr", "femto-gr", "fg", "femtog", "femto-g"], "followed_by": None},

    # Mass^-1
    "per_kg": {"dim": "mass", "degree": -1, "scale": 1e-3, "terms": ["kg-1", "kilog-1", "kilo-g-1", "kgr-1", "kilogr-1", "kilo-gr-1", "kgr⁻¹", "kilogr⁻¹", "kilo-gr⁻¹", "kg⁻¹", "kilog⁻¹", "kilo-g⁻¹"], "followed_by": None},
    "per_hg": {"dim": "mass", "degree": -1, "scale": 1e-2, "terms": ["hg-1", "hectog-1", "hecto-g-1", "hgr-1", "hectogr-1", "hecto-gr-1", "hgr⁻¹", "hectogr⁻¹", "hecto-gr⁻¹", "hg⁻¹", "hectog⁻¹", "hecto-g⁻¹"], "followed_by": None},
    "per_dag": {"dim": "mass", "degree": -1, "scale": 1e-1, "terms": ["dag-1", "decag-1", "deca-g-1", "dagr-1", "decagr-1", "deca-gr-1", "dagr⁻¹", "decagr⁻¹", "deca-gr⁻¹", "dag⁻¹", "decag⁻¹", "deca-g⁻¹"], "followed_by": None},
    "per_g": {"dim": "mass", "degree": -1, "scale": 1e0, "terms": ["g-1", "gr-1", "gr⁻¹", "g⁻¹"], "followed_by": None},
    "per_dg": {"dim": "mass", "degree": -1, "scale": 1e1, "terms": ["dg-1", "decig-1", "deci-g-1", "dgr-1", "decigr-1", "deci-gr-1", "dgr⁻¹", "decigr⁻¹", "deci-gr⁻¹", "dg⁻¹", "decig⁻¹", "deci-g⁻¹"], "followed_by": None},
    "per_cg": {"dim": "mass", "degree": -1, "scale": 1e2, "terms": ["cg-1", "centig-1", "centi-g-1", "cgr-1", "centigr-1", "centi-gr-1", "cgr⁻¹", "centigr⁻¹", "centi-gr⁻¹", "cg⁻¹", "centig⁻¹", "centi-g⁻¹"], "followed_by": None},
    "per_mg": {"dim": "mass", "degree": -1, "scale": 1e3, "terms": ["mg-1", "millig-1", "milli-g-1", "milig-1", "mili-g-1", "mgr-1", "milligr-1", "milli-gr-1", "miligr-1", "mili-gr-1", "mgr⁻¹", "milligr⁻¹", "milli-gr⁻¹", "miligr⁻¹", "mili-gr⁻¹", "mg⁻¹", "millig⁻¹", "milli-g⁻¹", "milig⁻¹", "mili-g⁻¹"], "followed_by": None},
    "per_μg": {"dim": "mass", "degree": -1, "scale": 1e6, "terms": ["μg-1", "ug-1", "µg-1", "microg-1", "micro-g-1", "μgr-1", "ugr-1", "µgr-1", "microgr-1", "micro-gr-1", "μgr⁻¹", "ugr⁻¹", "µgr⁻¹", "microgr⁻¹", "micro-gr⁻¹", "μg⁻¹", "ug⁻¹", "µg⁻¹", "microg⁻¹", "micro-g⁻¹"], "followed_by": None},
    "per_ng": {"dim": "mass", "degree": -1, "scale": 1e9, "terms": ["ng-1", "nanog-1", "nano-g-1", "ngr-1", "nanogr-1", "nano-gr-1", "ngr⁻¹", "nanogr⁻¹", "nano-gr⁻¹", "ng⁻¹", "nanog⁻¹", "nano-g⁻¹"], "followed_by": None},
    "per_pg": {"dim": "mass", "degree": -1, "scale": 1e12, "terms": ["pg-1", "picog-1", "pico-g-1", "pgr-1", "picogr-1", "pico-gr-1", "pgr⁻¹", "picogr⁻¹", "pico-gr⁻¹", "pg⁻¹", "picog⁻¹", "pico-g⁻¹"], "followed_by": None},
    "per_fg": {"dim": "mass", "degree": -1, "scale": 1e15, "terms": ["fg-1", "femtog-1", "femto-g-1", "fgr-1", "femtogr-1", "femto-gr-1", "fgr⁻¹", "femtogr⁻¹", "femto-gr⁻¹", "fg⁻¹", "femtog⁻¹", "femto-g⁻¹"], "followed_by": None},

    # Length
    "km":  {"dim": "length", "degree": 1, "scale": 1e5, "terms": ["kmetre", "kilometre", "kilo-metre", "kmetres", "kilometres", "kilo-metres", "km", "kilom", "kilo-m"], "followed_by": None},
    "hm":  {"dim": "length", "degree": 1, "scale": 1e4, "terms": ["hmetre", "hectometre", "hecto-metre", "hmetres", "hectometres", "hecto-metres", "hm", "hectom", "hecto-m"], "followed_by": None},
    "dam": {"dim": "length", "degree": 1, "scale": 1e3, "terms": ["dametre", "decametre", "deca-metre", "dametres", "decametres", "deca-metres", "dam", "decam", "deca-m"], "followed_by": None},
    "m":   {"dim": "length", "degree": 1, "scale": 1e2, "terms": ["metre", "metres", "m"], "followed_by": "cm"},
    "dm":  {"dim": "length", "degree": 1, "scale": 1e1, "terms": ["dmetre", "decimetre", "deci-metre", "dmetres", "decimetres", "deci-metres", "dm", "decim", "deci-m"], "followed_by": None},
    "cm":  {"dim": "length", "degree": 1, "scale": 1e0, "terms": ["cmetre", "centimetre", "centi-metre", "cmetres", "centimetres", "centi-metres", "cm", "centim", "centi-m"], "followed_by": None},
    "mm":  {"dim": "length", "degree": 1, "scale": 1e-1, "terms": ["mmetre", "millimetre", "milli-metre", "milimetre", "mili-metre", "mmetres", "millimetres", "milli-metres", "milimetres", "mili-metres", "mm", "millim", "milli-m", "milim", "mili-m"], "followed_by": None},
    "μm":  {"dim": "length", "degree": 1, "scale": 1e-4, "terms": ["μmetre", "umetre", "µmetre", "micrometre", "micro-metre", "μmetres", "umetres", "µmetres", "micrometres", "micro-metres", "μm", "um", "µm", "microm", "micro-m"], "followed_by": None},
    "nm":  {"dim": "length", "degree": 1, "scale": 1e-7, "terms": ["nmetre", "nanometre", "nano-metre", "nmetres", "nanometres", "nano-metres", "nm", "nanom", "nano-m"], "followed_by": None},
    "pm":  {"dim": "length", "degree": 1, "scale": 1e-10, "terms": ["pmetre", "picometre", "pico-metre", "pmetres", "picometres", "pico-metres", "pm", "picom", "pico-m"], "followed_by": None},
    "fm":  {"dim": "length", "degree": 1, "scale": 1e-13, "terms": ["fmetre", "femtometre", "femto-metre", "fmetres", "femtometres", "femto-metres", "fm", "femtom", "femto-m"], "followed_by": None},

    # Length^-1
    "per_km":  {"dim": "length", "degree": -1, "scale": 1e-5, "terms": ["km-1", "kilom-1", "kilo-m-1", "km⁻¹", "kilom⁻¹", "kilo-m⁻¹"], "followed_by": None},
    "per_hm":  {"dim": "length", "degree": -1, "scale": 1e-4, "terms": ["hm-1", "hectom-1", "hecto-m-1", "hm⁻¹", "hectom⁻¹", "hecto-m⁻¹"], "followed_by": None},
    "per_dam": {"dim": "length", "degree": -1, "scale": 1e-3, "terms": ["dam-1", "decam-1", "deca-m-1", "dam⁻¹", "decam⁻¹", "deca-m⁻¹"], "followed_by": None},
    "per_m":   {"dim": "length", "degree": -1, "scale": 1e-2, "terms": ["m-1", "m⁻¹"], "followed_by": None},
    "per_dm":  {"dim": "length", "degree": -1, "scale": 1e-1, "terms": ["dm-1", "decim-1", "deci-m-1", "dm⁻¹", "decim⁻¹", "deci-m⁻¹"], "followed_by": None},
    "per_cm":  {"dim": "length", "degree": -1, "scale": 1e0, "terms": ["cm-1", "centim-1", "centi-m-1", "cm⁻¹", "centim⁻¹", "centi-m⁻¹"], "followed_by": None},
    "per_mm":  {"dim": "length", "degree": -1, "scale": 1e1, "terms": ["mm-1", "millim-1", "milli-m-1", "milim-1", "mili-m-1", "mm⁻¹", "millim⁻¹", "milli-m⁻¹", "milim⁻¹", "mili-m⁻¹"], "followed_by": None},
    "per_μm":  {"dim": "length", "degree": -1, "scale": 1e4, "terms": ["μm-1", "um-1", "µm-1", "microm-1", "micro-m-1", "μm⁻¹", "um⁻¹", "µm⁻¹", "microm⁻¹", "micro-m⁻¹"], "followed_by": None},
    "per_nm":  {"dim": "length", "degree": -1, "scale": 1e7, "terms": ["nm-1", "nanom-1", "nano-m-1", "nm⁻¹", "nanom⁻¹", "nano-m⁻¹"], "followed_by": None},
    "per_pm":  {"dim": "length", "degree": -1, "scale": 1e10, "terms": ["pm-1", "picom-1", "pico-m-1", "pm⁻¹", "picom⁻¹", "pico-m⁻¹"], "followed_by": None},
    "per_fm":  {"dim": "length", "degree": -1, "scale": 1e13, "terms": ["fm-1", "femtom-1", "femto-m-1", "fm⁻¹", "femtom⁻¹", "femto-m⁻¹"], "followed_by": None},

    # Length^2
    "km2":  {"dim": "length", "degree": 2, "scale": 1e10, "terms": ["km2", "kilom2", "kilo-m2", "km²", "kilom²", "kilo-m²"], "followed_by": None},
    "hm2":  {"dim": "length", "degree": 2, "scale": 1e8, "terms": ["hm2", "hectom2", "hecto-m2", "hm²", "hectom²", "hecto-m²"], "followed_by": None},
    "dam2": {"dim": "length", "degree": 2, "scale": 1e6, "terms": ["dam2", "decam2", "deca-m2", "dam²", "decam²", "deca-m²"], "followed_by": None},
    "m2":   {"dim": "length", "degree": 2, "scale": 1e4, "terms": ["m2", "m²"], "followed_by": None},
    "dm2":  {"dim": "length", "degree": 2, "scale": 1e2, "terms": ["dm2", "decim2", "deci-m2", "dm²", "decim²", "deci-m²"], "followed_by": None},
    "cm2":  {"dim": "length", "degree": 2, "scale": 1e0, "terms": ["cm2", "centim2", "centi-m2", "cm²", "centim²", "centi-m²"], "followed_by": None},
    "mm2":  {"dim": "length", "degree": 2, "scale": 1e-2, "terms": ["mm2", "millim2", "milli-m2", "milim2", "mili-m2", "mm²", "millim²", "milli-m²", "milim²", "mili-m²"], "followed_by": None},
    "μm2":  {"dim": "length", "degree": 2, "scale": 1e-8, "terms": ["μm2", "um2", "µm2", "microm2", "micro-m2", "μm²", "um²", "µm²", "microm²", "micro-m²"], "followed_by": None},
    "nm2":  {"dim": "length", "degree": 2, "scale": 1e-14, "terms": ["nm2", "nanom2", "nano-m2", "nm²", "nanom²", "nano-m²"], "followed_by": None},
    "pm2":  {"dim": "length", "degree": 2, "scale": 1e-20, "terms": ["pm2", "picom2", "pico-m2", "pm²", "picom²", "pico-m²"], "followed_by": None},
    "fm2":  {"dim": "length", "degree": 2, "scale": 1e-26, "terms": ["fm2", "femtom2", "femto-m2", "fm²", "femtom²", "femto-m²"], "followed_by": None},

    # Length^-2
    "per_km2":  {"dim": "length", "degree": -2, "scale": 1e-10, "terms": ["km-2", "kilom-2", "kilo-m-2", "km⁻²", "kilom⁻²", "kilo-m⁻²"], "followed_by": None},
    "per_hm2":  {"dim": "length", "degree": -2, "scale": 1e-8, "terms": ["hm-2", "hectom-2", "hecto-m-2", "hm⁻²", "hectom⁻²", "hecto-m⁻²"], "followed_by": None},
    "per_dam2": {"dim": "length", "degree": -2, "scale": 1e-6, "terms": ["dam-2", "decam-2", "deca-m-2", "dam⁻²", "decam⁻²", "deca-m⁻²"], "followed_by": None},
    "per_m2":   {"dim": "length", "degree": -2, "scale": 1e-4, "terms": ["m-2", "m⁻²"], "followed_by": None},
    "per_dm2":  {"dim": "length", "degree": -2, "scale": 1e-2, "terms": ["dm-2", "decim-2", "deci-m-2", "dm⁻²", "decim⁻²", "deci-m⁻²"], "followed_by": None},
    "per_cm2":  {"dim": "length", "degree": -2, "scale": 1e0, "terms": ["cm-2", "centim-2", "centi-m-2", "cm⁻²", "centim⁻²", "centi-m⁻²"], "followed_by": None},
    "per_mm2":  {"dim": "length", "degree": -2, "scale": 1e2, "terms": ["mm-2", "millim-2", "milli-m-2", "milim-2", "mili-m-2", "mm⁻²", "millim⁻²", "milli-m⁻²", "milim⁻²", "mili-m⁻²"], "followed_by": None},
    "per_μm2":  {"dim": "length", "degree": -2, "scale": 1e8, "terms": ["μm-2", "um-2", "µm-2", "microm-2", "micro-m-2", "μm⁻²", "um⁻²", "µm⁻²", "microm⁻²", "micro-m⁻²"], "followed_by": None},
    "per_nm2":  {"dim": "length", "degree": -2, "scale": 1e14, "terms": ["nm-2", "nanom-2", "nano-m-2", "nm⁻²", "nanom⁻²", "nano-m⁻²"], "followed_by": None},
    "per_pm2":  {"dim": "length", "degree": -2, "scale": 1e20, "terms": ["pm-2", "picom-2", "pico-m-2", "pm⁻²", "picom⁻²", "pico-m⁻²"], "followed_by": None},
    "per_fm2":  {"dim": "length", "degree": -2, "scale": 1e26, "terms": ["fm-2", "femtom-2", "femto-m-2", "fm⁻²", "femtom⁻²", "femto-m⁻²"], "followed_by": None},

    # Length^3
    "km3":    {"dim": "length", "degree": 3, "scale": 1e15, "terms": ["km3", "kilom3", "kilo-m3", "km³", "kilom³", "kilo-m³"], "followed_by": None},
    "hm3":    {"dim": "length", "degree": 3, "scale": 1e12, "terms": ["hm3", "hectom3", "hecto-m3", "hm³", "hectom³", "hecto-m³"], "followed_by": None},
    "dam3":   {"dim": "length", "degree": 3, "scale": 1e9, "terms": ["dam3", "decam3", "deca-m3", "dam³", "decam³", "deca-m³"], "followed_by": None},
    "m3":     {"dim": "length", "degree": 3, "scale": 1e6, "terms": ["m3", "m³"], "followed_by": None},
    "dm3":    {"dim": "length", "degree": 3, "scale": 1e3, "terms": ["dm3", "decim3", "deci-m3", "dm³", "decim³", "deci-m³"], "followed_by": None},
    "cm3":    {"dim": "length", "degree": 3, "scale": 1e0, "terms": ["cm3", "centim3", "centi-m3", "cm³", "centim³", "centi-m³"], "followed_by": None},
    "mm3":    {"dim": "length", "degree": 3, "scale": 1e-3, "terms": ["mm3", "millim3", "milli-m3", "milim3", "mili-m3", "mm³", "millim³", "milli-m³", "milim³", "mili-m³"], "followed_by": None},
    "μm3":    {"dim": "length", "degree": 3, "scale": 1e-12, "terms": ["μm3", "um3", "µm3", "microm3", "micro-m3", "μm³", "um³", "µm³", "microm³", "micro-m³"], "followed_by": None},
    "nm3":    {"dim": "length", "degree": 3, "scale": 1e-21, "terms": ["nm3", "nanom3", "nano-m3", "nm³", "nanom³", "nano-m³"], "followed_by": None},
    "pm3":    {"dim": "length", "degree": 3, "scale": 1e-30, "terms": ["pm3", "picom3", "pico-m3", "pm³", "picom³", "pico-m³"], "followed_by": None},
    "fm3":    {"dim": "length", "degree": 3, "scale": 1e-39, "terms": ["fm3", "femtom3", "femto-m3", "fm³", "femtom³", "femto-m³"], "followed_by": None},
    "cac":    {"dim": "length", "degree": 3, "scale": 5.0, "terms": ["cac", "c.a.c", "cuillere à café", "cuillères à café"], "followed_by": None},
    "goutte": {"dim": "length", "degree": 3, "scale": 0.05, "terms": ["gt", "goutte", "gouttes"], "followed_by": None},
    "µl":     {"dim": "length", "degree": 3, "scale": 1e-3, "terms": ["µl", "μl", "µlitre", "μlitres", "micro litre", "micro litres", "microlitre", "microlitres", "micro-litre", "micro-litres"], "followed_by": None},
    "ml":     {"dim": "length", "degree": 3, "scale": 1e0, "terms": ["mililitre", "millilitre", "mililitres", "millilitres", "ml"], "followed_by": None},
    "cl":     {"dim": "length", "degree": 3, "scale": 1e1, "terms": ["centilitre", "centilitres", "cl"], "followed_by": None},
    "dl":     {"dim": "length", "degree": 3, "scale": 1e2, "terms": ["decilitre", "decilitres", "dl"], "followed_by": None},
    "l":      {"dim": "length", "degree": 3, "scale": 1e3, "terms": ["litre", "litres", "l", "dm3"], "followed_by": "ml"},

    # Length^-3
    "per_km3":  {"dim": "length", "degree": -3, "scale": 1e-15, "terms": ["km-3", "kilom-3", "kilo-m-3", "km⁻³", "kilom⁻³", "kilo-m⁻³"], "followed_by": None},
    "per_hm3":  {"dim": "length", "degree": -3, "scale": 1e-12, "terms": ["hm-3", "hectom-3", "hecto-m-3", "hm⁻³", "hectom⁻³", "hecto-m⁻³"], "followed_by": None},
    "per_dam3": {"dim": "length", "degree": -3, "scale": 1e-9, "terms": ["dam-3", "decam-3", "deca-m-3", "dam⁻³", "decam⁻³", "deca-m⁻³"], "followed_by": None},
    "per_dm3":  {"dim": "length", "degree": -3, "scale": 1e-6, "terms": ["dm-3", "decim-3", "deci-m-3", "dm⁻³", "decim⁻³", "deci-m⁻³"], "followed_by": None},
    "per_m3":   {"dim": "length", "degree": -3, "scale": 1e-3, "terms": ["m-3", "m⁻³"], "followed_by": None},
    "per_cm3":  {"dim": "length", "degree": -3, "scale": 1e0, "terms": ["cm-3", "centim-3", "centi-m-3", "cm⁻³", "centim⁻³", "centi-m⁻³"], "followed_by": None},
    "per_mm3":  {"dim": "length", "degree": -3, "scale": 1e3, "terms": ["mm-3", "millim-3", "milli-m-3", "milim-3", "mili-m-3", "mm⁻³", "millim⁻³", "milli-m⁻³", "milim⁻³", "mili-m⁻³"], "followed_by": None},
    "per_μm3":  {"dim": "length", "degree": -3, "scale": 1e12, "terms": ["μm-3", "um-3", "µm-3", "microm-3", "micro-m-3", "μm⁻³", "um⁻³", "µm⁻³", "microm⁻³", "micro-m⁻³"], "followed_by": None},
    "per_nm3":  {"dim": "length", "degree": -3, "scale": 1e21, "terms": ["nm-3", "nanom-3", "nano-m-3", "nm⁻³", "nanom⁻³", "nano-m⁻³"], "followed_by": None},
    "per_pm3":  {"dim": "length", "degree": -3, "scale": 1e30, "terms": ["pm-3", "picom-3", "pico-m-3", "pm⁻³", "picom⁻³", "pico-m⁻³"], "followed_by": None},
    "per_fm3":  {"dim": "length", "degree": -3, "scale": 1e39, "terms": ["fm-3", "femtom-3", "femto-m-3", "fm⁻³", "femtom⁻³", "femto-m⁻³"], "followed_by": None},

    # Mole
    "kmol":  {"dim": "mole", "degree": 1, "scale": 1e3, "terms": ["kmol", "kilomol", "kilo-mol", "kmole", "kilomole", "kilo-mole", "kmoles", "kilomoles", "kilo-moles"], "followed_by": None},
    "hmol":  {"dim": "mole", "degree": 1, "scale": 1e2, "terms": ["hmol", "hectomol", "hecto-mol", "hmole", "hectomole", "hecto-mole", "hmoles", "hectomoles", "hecto-moles"], "followed_by": None},
    "damol": {"dim": "mole", "degree": 1, "scale": 1e1, "terms": ["damol", "decamol", "deca-mol", "damole", "decamole", "deca-mole", "damoles", "decamoles", "deca-moles"], "followed_by": None},
    "mol":   {"dim": "mole", "degree": 1, "scale": 1e0,  "terms": ["mol", "mole", "moles"], "followed_by": None},
    "dmol":  {"dim": "mole", "degree": 1, "scale": 1e-1, "terms": ["dmol", "decimol", "deci-mol", "dmole", "decimole", "deci-mole", "dmoles", "decimoles", "deci-moles"], "followed_by": None},
    "cmol":  {"dim": "mole", "degree": 1, "scale": 1e-2, "terms": ["cmol", "centimol", "centi-mol", "cmole", "centimole", "centi-mole", "cmoles", "centimoles", "centi-moles"], "followed_by": None},
    "mmol":  {"dim": "mole", "degree": 1, "scale": 1e-3, "terms": ["mmol", "millimol", "milli-mol", "milimol", "mili-mol", "mmole", "millimole", "milli-mole", "milimole", "mili-mole", "mmoles", "millimoles", "milli-moles", "milimoles", "mili-moles"], "followed_by": None},
    "μmol":  {"dim": "mole", "degree": 1, "scale": 1e-6, "terms": ["μmol", "umol", "µmol", "micromol", "micro-mol", "μmole", "umole", "µmole", "micromole", "micro-mole", "μmoles", "umoles", "µmoles", "micromoles", "micro-moles"], "followed_by": None},
    "nmol":  {"dim": "mole", "degree": 1, "scale": 1e-9, "terms": ["nmol", "nanomol", "nano-mol", "nmole", "nanomole", "nano-mole", "nmoles", "nanomoles", "nano-moles"], "followed_by": None},
    "pmol":  {"dim": "mole", "degree": 1, "scale": 1e-12, "terms": ["pmol", "picomol", "pico-mol", "pmole", "picomole", "pico-mole", "pmoles", "picomoles", "pico-moles"], "followed_by": None},
    "fmol":  {"dim": "mole", "degree": 1, "scale": 1e-15, "terms": ["fmol", "femtomol", "femto-mol", "fmole", "femtomole", "femto-mole", "fmoles", "femtomoles", "femto-moles"], "followed_by": None},

    # UI
    "kui":  {"dim": "ui", "degree": 1, "scale": 1e3, "terms": ["kui", "kiloui", "kilo-ui", "ku", "kilou", "kilo-u"], "followed_by": None},
    "hui":  {"dim": "ui", "degree": 1, "scale": 1e2, "terms": ["hui", "hectoui", "hecto-ui", "hu", "hectou", "hecto-u"], "followed_by": None},
    "daui": {"dim": "ui", "degree": 1, "scale": 1e1, "terms": ["daui", "decaui", "deca-ui", "dau", "decau", "deca-u"], "followed_by": None},
    "ui":   {"dim": "ui", "degree": 1, "scale": 1e0, "terms": ["ui", "u"], "followed_by": None},
    "dui":  {"dim": "ui", "degree": 1, "scale": 1e-1, "terms": ["dui", "deciui", "deci-ui", "du", "deciu", "deci-u"], "followed_by": None},
    "cui":  {"dim": "ui", "degree": 1, "scale": 1e-2, "terms": ["cui", "centiui", "centi-ui", "cu", "centiu", "centi-u"], "followed_by": None},
    "mui":  {"dim": "ui", "degree": 1, "scale": 1e-3, "terms": ["mui", "milliui", "milli-ui", "miliui", "mili-ui", "mu", "milliu", "milli-u", "miliu", "mili-u"], "followed_by": None},
    "μui":  {"dim": "ui", "degree": 1, "scale": 1e-6, "terms": ["μui", "uui", "µui", "microui", "micro-ui", "μu", "uu", "µu", "microu", "micro-u"], "followed_by": None},
    "nui":  {"dim": "ui", "degree": 1, "scale": 1e-9, "terms": ["nui", "nanoui", "nano-ui", "nu", "nanou", "nano-u"], "followed_by": None},
    "pui":  {"dim": "ui", "degree": 1, "scale": 1e-12, "terms": ["pui", "picoui", "pico-ui", "pu", "picou", "pico-u"], "followed_by": None},
    "fui":  {"dim": "ui", "degree": 1, "scale": 1e-15, "terms": ["fui", "femtoui", "femto-ui", "fu", "femtou", "femto-u"], "followed_by": None},

    # UI^-1
    "per_kui":  {"dim": "ui", "degree": -1, "scale": 1e-3, "terms": ["kui-1", "kiloui-1", "kilo-ui-1", "kui⁻¹", "kiloui⁻¹", "kilo-ui⁻¹"], "followed_by": None},
    "per_hui":  {"dim": "ui", "degree": -1, "scale": 1e-2, "terms": ["hui-1", "hectoui-1", "hecto-ui-1", "hui⁻¹", "hectoui⁻¹", "hecto-ui⁻¹"], "followed_by": None},
    "per_daui": {"dim": "ui", "degree": -1, "scale": 1e-1, "terms": ["daui-1", "decaui-1", "deca-ui-1", "daui⁻¹", "decaui⁻¹", "deca-ui⁻¹"], "followed_by": None},
    "per_ui":   {"dim": "ui", "degree": -1, "scale": 1e0, "terms": ["ui-1", "ui⁻¹"], "followed_by": None},
    "per_dui":  {"dim": "ui", "degree": -1, "scale": 1e1, "terms": ["dui-1", "deciui-1", "deci-ui-1", "dui⁻¹", "deciui⁻¹", "deci-ui⁻¹"], "followed_by": None},
    "per_cui":  {"dim": "ui", "degree": -1, "scale": 1e2, "terms": ["cui-1", "centiui-1", "centi-ui-1", "cui⁻¹", "centiui⁻¹", "centi-ui⁻¹"], "followed_by": None},
    "per_mui":  {"dim": "ui", "degree": -1, "scale": 1e3, "terms": ["mui-1", "milliui-1", "milli-ui-1", "miliui-1", "mili-ui-1", "mui⁻¹", "milliui⁻¹", "milli-ui⁻¹", "miliui⁻¹", "mili-ui⁻¹"], "followed_by": None},
    "per_μui":  {"dim": "ui", "degree": -1, "scale": 1e6, "terms": ["μui-1", "uui-1", "µui-1", "microui-1", "micro-ui-1", "μui⁻¹", "uui⁻¹", "µui⁻¹", "microui⁻¹", "micro-ui⁻¹"], "followed_by": None},
    "per_nui":  {"dim": "ui", "degree": -1, "scale": 1e9, "terms": ["nui-1", "nanoui-1", "nano-ui-1", "nui⁻¹", "nanoui⁻¹", "nano-ui⁻¹"], "followed_by": None},
    "per_pui":  {"dim": "ui", "degree": -1, "scale": 1e12, "terms": ["pui-1", "picoui-1", "pico-ui-1", "pui⁻¹", "picoui⁻¹", "pico-ui⁻¹"], "followed_by": None},
    "per_fui":  {"dim": "ui", "degree": -1, "scale": 1e15, "terms": ["fui-1", "femtoui-1", "femto-ui-1", "fui⁻¹", "femtoui⁻¹", "femto-ui⁻¹"], "followed_by": None},

    # Pressure
    "kPa":  {"dim": "pressure", "degree": 1, "scale": 1e3, "terms": ["kPa", "kiloPa", "kilo-Pa"], "followed_by": None},
    "hPa":  {"dim": "pressure", "degree": 1, "scale": 1e2, "terms": ["hPa", "hectoPa", "hecto-Pa"], "followed_by": None},
    "daPa": {"dim": "pressure", "degree": 1, "scale": 1e1, "terms": ["daPa", "decaPa", "deca-Pa"], "followed_by": None},
    "Pa":   {"dim": "pressure", "degree": 1, "scale": 1e0, "terms": ["Pa"], "followed_by": None},
    "dPa":  {"dim": "pressure", "degree": 1, "scale": 1e-1, "terms": ["dPa", "deciPa", "deci-Pa"], "followed_by": None},
    "cPa":  {"dim": "pressure", "degree": 1, "scale": 1e-2, "terms": ["cPa", "centiPa", "centi-Pa"], "followed_by": None},
    "mPa":  {"dim": "pressure", "degree": 1, "scale": 1e-3, "terms": ["mPa", "milliPa", "milli-Pa", "miliPa", "mili-Pa"], "followed_by": None},
    "μPa":  {"dim": "pressure", "degree": 1, "scale": 1e-6, "terms": ["μPa", "uPa", "µPa", "microPa", "micro-Pa"], "followed_by": None},
    "nPa":  {"dim": "pressure", "degree": 1, "scale": 1e-9, "terms": ["nPa", "nanoPa", "nano-Pa"], "followed_by": None},
    "pPa":  {"dim": "pressure", "degree": 1, "scale": 1e-12, "terms": ["pPa", "picoPa", "pico-Pa"], "followed_by": None},
    "fPa":  {"dim": "pressure", "degree": 1, "scale": 1e-15, "terms": ["fPa", "femtoPa", "femto-Pa"], "followed_by": None},
    "mmHg": {"dim": "pressure", "degree": 1, "scale": 133, "terms": ["mmHg"], "followed_by": None},

    # Percent (special unit)
    "%": {"dim": "%", "degree": 1, "scale": 1, "terms": ["%"], "followed_by": None},

    # Logarithm (special unit)
    "log": {"dim": "log", "degree": 1, "scale": 1, "terms": ["log"], "followed_by": None},

    # Time
    "year":       {"dim": "time", "degree": 1, "scale": 31557600.0, "terms": ["an", "année", "ans", "années"], "followed_by": None},
    "month":      {"dim": "time", "degree": 1, "scale": 2628002.88, "terms": ["mois"], "followed_by": None},
    "week":       {"dim": "time", "degree": 1, "scale": 604800, "terms": ["semaine", "semaines", "sem"], "followed_by": None},
    "day":        {"dim": "time", "degree": 1, "scale": 86400, "terms": ["jour", "jours", "j"], "followed_by": None},
    "hour":       {"dim": "time", "degree": 1, "scale": 3600, "terms": ["heure", "heures", "h"], "followed_by": "minute"},
    "minute":     {"dim": "time", "degree": 1, "scale": 60, "terms": ["mn", "min", "minute", "minutes"], "followed_by": "second"},
    "second":     {"dim": "time", "degree": 1, "scale": 1, "terms": ["seconde", "secondes", "s"], "followed_by": None},
    "arc-minute": {"dim": "time", "degree": 1, "scale": 60, "terms": ["'"], "followed_by": "arc-second"},
    "arc-second": {"dim": "time", "degree": 1, "scale": 1, "terms": ['"', "''"], "followed_by": None},
    "degree":     {"dim": "time", "degree": 1, "scale": 3600, "terms": ["degre", "°", "deg"], "followed_by": "arc-minute"},

    # Temperature
    "celsius": {"dim": "temperature", "degree": 1, "scale": 1, "terms": ["°C", "° celsius", "celsius"], "followed_by": None},

    # Count
    "x10*1":  {"dim": "count", "degree": 0, "scale": 1e1, "terms": ["x10*1"], "followed_by": None},
    "x10*2":  {"dim": "count", "degree": 0, "scale": 1e2, "terms": ["x10*2"], "followed_by": None},
    "x10*3":  {"dim": "count", "degree": 0, "scale": 1e3, "terms": ["x10*3"], "followed_by": None},
    "x10*4":  {"dim": "count", "degree": 0, "scale": 1e4, "terms": ["x10*4"], "followed_by": None},
    "x10*5":  {"dim": "count", "degree": 0, "scale": 1e5, "terms": ["x10*5"], "followed_by": None},
    "x10*6":  {"dim": "count", "degree": 0, "scale": 1e6, "terms": ["x10*6"], "followed_by": None},
    "x10*7":  {"dim": "count", "degree": 0, "scale": 1e7, "terms": ["x10*7"], "followed_by": None},
    "x10*8":  {"dim": "count", "degree": 0, "scale": 1e8, "terms": ["x10*8"], "followed_by": None},
    "x10*9":  {"dim": "count", "degree": 0, "scale": 1e9, "terms": ["x10*9"], "followed_by": None},
    "x10*10": {"dim": "count", "degree": 0, "scale": 1e10, "terms": ["x10*10"], "followed_by": None},
    "x10*11": {"dim": "count", "degree": 0, "scale": 1e11, "terms": ["x10*11"], "followed_by": None},
    "x10*12": {"dim": "count", "degree": 0, "scale": 1e12, "terms": ["x10*12"], "followed_by": None},
    "x10*13": {"dim": "count", "degree": 0, "scale": 1e13, "terms": ["x10*13"], "followed_by": None},
    "x10*14": {"dim": "count", "degree": 0, "scale": 1e14, "terms": ["x10*14"], "followed_by": None},
    "x10*15": {"dim": "count", "degree": 0, "scale": 1e15, "terms": ["x10*15"], "followed_by": None},
}
# fmt: on
