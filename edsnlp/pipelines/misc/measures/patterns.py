import spacy

from edsnlp.pipelines.misc.measures.measures import (
    CompositeMeasure,
    SimpleMeasure,
    make_multi_getter,
    make_simple_getter,
)


class CompositeSize(CompositeMeasure):
    """
    Composite size measure. Supports the following units:
    - mm
    - cm
    - dm
    - m
    """

    mm = property(make_multi_getter("mm"))
    cm = property(make_multi_getter("cm"))
    dm = property(make_multi_getter("dm"))
    m = property(make_multi_getter("m"))


@spacy.registry.misc("eds.measures.size")
class Size(SimpleMeasure):
    """
    Size measure. Supports the following units:
    - mm
    - cm
    - dm
    - m
    """

    COMPOSITE = CompositeSize
    UNITS = {
        "mm": {"prefix": "mill?im", "abbr": "mm", "value": 1},
        "cm": {"prefix": "centim", "abbr": "cm", "value": 10},
        "dm": {"prefix": "decim", "abbr": "dm", "value": 100},
        "m": {"prefix": "metre", "abbr": "m", "value": 1000},
    }

    @classmethod
    def parse(cls, int_part, dec_part, unit, infix=False):
        result = float("{}.{}".format(int_part, dec_part))
        return cls(result, unit)

    mm = property(make_simple_getter("mm"))
    cm = property(make_simple_getter("cm"))
    dm = property(make_simple_getter("dm"))
    m = property(make_simple_getter("m"))


@spacy.registry.misc("eds.measures.weight")
class Weight(SimpleMeasure):
    """
    Weight measure. Supports the following units:
    - mg
    - cg
    - dg
    - g
    - kg
    """

    COMPOSITE = None
    UNITS = {
        "mg": {"prefix": "mill?ig", "abbr": "mg", "value": 1},
        "cg": {"prefix": "centig", "abbr": "cg", "value": 10},
        "dg": {"prefix": "decig", "abbr": "dg", "value": 100},
        "g": {"prefix": "gram", "abbr": "g", "value": 1000},
        "kg": {"prefix": "kilo", "abbr": "kg", "value": 1000000},
    }

    @classmethod
    def parse(cls, int_part, dec_part, unit, infix=False):
        result = float("{}.{}".format(int_part, dec_part))
        return cls(result, unit)

    mg = property(make_simple_getter("mg"))
    cg = property(make_simple_getter("cg"))
    dg = property(make_simple_getter("dg"))
    g = property(make_simple_getter("g"))
    kg = property(make_simple_getter("kg"))


@spacy.registry.misc("eds.measures.angle")
class Angle(SimpleMeasure):
    """
    Angle measure. Supports the following units:
    - h
    - deg
    """

    COMPOSITE = None
    UNITS = {
        "h": {"prefix": "heur", "abbr": "h", "value": 1},
        "degree": {"prefix": "deg", "abbr": "°", "value": 1}, 
    }

    @classmethod
    def parse(cls, int_part, dec_part, unit, infix=False):
        if infix:
            result = float(int_part) + int(dec_part) / 60.0
            return cls(result, unit)
        result = float("{}.{}".format(int_part, dec_part))
        return cls(result, unit)

    h = property(make_simple_getter("h"))
    degree = property(make_simple_getter("degree"))

@spacy.registry.misc("eds.measures.temperature")
class Temperature(SimpleMeasure):
    """
    Temperature measure. Supports the following units:
    - CelsiusDegree

    Note that temperature and angle can be expressed usint the same symbol °
    But Celsius-degree is usually given in decimal units while angular degree 
    is usually given in degree and minutes. 
    """

    COMPOSITE = None
    UNITS = {
        "CelsiusDegree": {"prefix": "deg", "abbr": "°", "value": 1}, 
    }

    @classmethod
    def parse(cls, int_part, dec_part, unit, infix=False):
        result = float("{}.{}".format(int_part, dec_part))
        return cls(result, unit)

    CelsiusDegree = property(make_simple_getter("CelsiusDegree"))


@spacy.registry.misc("eds.measures.volume")
class Volume(SimpleMeasure):
    """
    Volume measure. Supports the following units:
    - mL
    - cL
    - dL
    - L
    - cc
    - goutte
    """

    COMPOSITE = None
    UNITS = {
        "mL": {"prefix": "mill?ilitre", "abbr": "ml", "value": 1},
        "cL": {"prefix": "centilitre", "abbr": "cl", "value": 10},
        "dL": {"prefix": "decilitre", "abbr": "dl", "value": 100},
        "L": {"prefix": "litre", "abbr": "l", "value": 1000},
        "goutte": {"prefix": "goutte?", "abbr": "gt", "value": 1},
        "cc": {"prefix": "cc", "abbr": "cc", "value": 10},
    }

    @classmethod
    def parse(cls, int_part, dec_part, unit, infix=False):
        result = float("{}.{}".format(int_part, dec_part))
        return cls(result, unit)

    mL = property(make_simple_getter("mL"))
    cL = property(make_simple_getter("cL"))
    dL = property(make_simple_getter("dL"))
    L = property(make_simple_getter("L"))
    cc = property(make_simple_getter("cc"))
    goutte = property(make_simple_getter("goutte"))

@spacy.registry.misc("eds.measures.energy")
class Energy(SimpleMeasure):
    """
    Energy measure. Supports the following units:
    - J
    - kJ
    - MJ
    - cal
    - kcal
    """

    COMPOSITE = None
    UNITS = {
        "J": {"prefix": "joule", "abbr": "j", "value": 1},
        "kJ": {"prefix": "kilojou", "abbr": "kj", "value": 0.001},
        "MJ": {"prefix": "megajou", "abbr": "mj", "value": 0.000001},
        "cal": {"prefix": "calorie", "abbr": "cal", "value": 4.184},
        "kcal":{"prefix": "kilocal",  "abbr": "kcal", "value": 4184}, 
    }

    @classmethod
    def parse(cls, int_part, dec_part, unit, infix=False):
        result = float("{}.{}".format(int_part, dec_part))
        return cls(result, unit)

    J = property(make_simple_getter("J"))
    kJ = property(make_simple_getter("kJ"))
    MJ = property(make_simple_getter("MJ"))
    cal = property(make_simple_getter("cal"))
    kcal = property(make_simple_getter("kcal"))

@spacy.registry.misc("eds.measures.pressure")
class Pressure(SimpleMeasure):
    """
    Pressure measure. Supports the following units:
    - Pa
    - kPa
    - hPa
    - atm
    - mmHg
    - Torr
    - bar
    - mbar
    - psi
    """

    COMPOSITE = None
    UNITS = {
        "Pa": {"prefix": "pascal", "abbr": "pa", "value": 1},
        "kPa": {"prefix": "kilopascal", "abbr": "kpa", "value": 1000},
        "hPa": {"prefix": "hectopascal", "abbr": "hpa", "value": 100},
        "atm": {"prefix": "atmosphere", "abbr": "atm", "value": 101325},
        "mmHg": {"prefix": "milli?mercure",  "abbr": "mmhg", "value": 133.3224}, 
        "Torr": {"prefix": "torr",  "abbr": "torr", "value": 133.3224}, 
        "bar": {"prefix": "bar",  "abbr": "bar", "value": 100000}, 
        "mbar": {"prefix": "millibar",  "abbr": "mbar", "value": 100}, 
        "psi": {"prefix": "psi",  "abbr": "psi", "value": 6894.757}, 
    }

    @classmethod
    def parse(cls, int_part, dec_part, unit, infix=False):
        result = float("{}.{}".format(int_part, dec_part))
        return cls(result, unit)

    Pa = property(make_simple_getter("Pa"))
    kPa = property(make_simple_getter("kPa"))
    hPa = property(make_simple_getter("hPa"))
    atm = property(make_simple_getter("atm"))
    mmHg = property(make_simple_getter("mmHg"))
    Torr = property(make_simple_getter("Torr"))
    bar = property(make_simple_getter("bar"))
    mbar = property(make_simple_getter("mbar"))
    psi = property(make_simple_getter("psi"))