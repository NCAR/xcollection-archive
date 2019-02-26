"""Top-level package for xcollection."""
from ._version import get_versions
from .config import get_options, set_options
from .core import analyzed_collection, operator

__version__ = get_versions()['version']
del get_versions
