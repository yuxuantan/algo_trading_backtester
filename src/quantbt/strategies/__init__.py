"""Built-in strategy modules.

Keep package init import-free to avoid circular/partial initialization issues
during dynamic strategy discovery.
"""

from pkgutil import iter_modules

__all__ = [m.name for m in iter_modules(__path__) if not m.name.startswith("_")]
