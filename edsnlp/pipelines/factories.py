# flake8: noqa: F811
from .core.advanced import factory
from .core.endlines import factory
from .core.matcher import factory
from .core.normalizer import factory
from .core.normalizer.accents import factory
from .core.normalizer.lowercase import factory
from .core.normalizer.pollution import factory
from .core.normalizer.quotes import factory
from .core.sentences import factory
from .misc.consultation_dates import factory
from .misc.dates import factory
from .misc.reason import factory
from .misc.sections import factory
from .ner.scores import charlson_factory, factory, sofa_factory
