"""edaio - Lightning-fast DataFrame introspection."""

from .core import glance
from .report import GlanceReport

__version__ = "0.5.0"
__all__ = ["glance", "GlanceReport"]