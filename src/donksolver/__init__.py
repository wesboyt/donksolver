"""donksolver: a tabular Monte Carlo CFR solver for no-limit hold'em.

    from donksolver import Abstraction, Solver, Solve, Bundle, preset, toy

The native core is a shared library with a C ABI, loaded with ctypes.
"""
from ._native import version as native_version, NativeError
from .config import Game, Solve, preset, toy
from .solver import Abstraction, Solver, tree_info
from .policy import Bundle, write_manifest

__version__ = "0.1.0"
__all__ = ["Abstraction", "Solver", "Bundle", "Game", "Solve", "preset", "toy", "tree_info",
           "write_manifest", "native_version", "NativeError", "__version__"]
