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


class CompositeWeight(CompositeMeasure):
    """
    Composite weight measure. Supports the following units:
    - mg
    - cg
    """

    mg = property(make_multi_getter("mg"))
    cg = property(make_multi_getter("cg"))


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

    COMPOSITE = CompositeWeight
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
        "deg": {"prefix": "deg", "abbr": "°", "value": 1}, 
    }

    @classmethod
    def parse(cls, int_part, dec_part, unit, infix=False):
        if infix:
            result = float(int_part) + int(dec_part) / 60.0
            return cls(result, unit)
        result = float("{}.{}".format(int_part, dec_part))
        return cls(result, unit)

    h = property(make_simple_getter("h"))


class CompositeVolume(CompositeMeasure):
    """
    Composite size measure. Supports the following units:
    - mL
    - cL
    - dL
    - L
    - cc
    - goutte
    """

    mL = property(make_multi_getter("mL"))
    cL = property(make_multi_getter("cL"))
    dL = property(make_multi_getter("dL"))
    L = property(make_multi_getter("L"))
    cc = property(make_multi_getter("cc"))
    goutte = property(make_multi_getter("goutte"))


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

    COMPOSITE = CompositeSize
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
