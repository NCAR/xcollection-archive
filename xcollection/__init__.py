"""Top-level package for xcollection."""
from .core import analyzed_collection, operator
from .config import set_options, get_options


from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
