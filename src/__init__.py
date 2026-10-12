"""AI photo enhancer package.

Subpackages are intentionally not imported eagerly: importing the lightweight
photo editor should not initialize the complete SAC training stack.
"""

from importlib import import_module

__all__ = ["envs", "sac"]


def __getattr__(name: str):
    if name in __all__:
        return import_module(f"src.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
