"""Unmochan public API.

The implementation namespace remains :mod:`unfoldlab` for backward
compatibility.  New code may import :mod:`unmochan`; existing ``unfoldlab``
imports continue to work.
"""

from unfoldlab import *  # noqa: F403
from unfoldlab import __all__ as __all__
from unfoldlab import __version__ as __version__
