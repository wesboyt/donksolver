"""Small solved policies for the serving audits (cached under artifacts/test/).

These are MECHANICS fixtures, not strong strategies: short solves whose only
job is to give the server non-trivial tables to read.
"""
import os
import time

import _common as T
from donksolver import Solver, Solve, preset, write_manifest

OUT = os.path.join(T.ARTIFACTS, "test")


def policy(name, stack_bb, iterations, threads=None):
    d = os.path.join(OUT, "%s_%dbb" % (name, stack_bb))
    pol = os.path.join(d, "policy.dpol")
    man = os.path.join(d, "bundle.json")
    game = preset(name, stack_bb)
    solve = Solve(algo="es", avg_mode="sampled", p_avg=0.05, seed=7)
    if not os.path.exists(man):
        s = Solver.nlhe(game, T.abstraction_path(), solve)
        t0 = time.time()
        s.solve(d, iterations, threads=threads or T.THREADS, chunk=max(iterations // 5, 1000),
                log=lambda m: print("  [fixture %s %dbb] %s" % (name, stack_bb, m)))
        print("  [fixture] solved %s %dbb in %.1fs" % (name, stack_bb, time.time() - t0))
        write_manifest(man, T.abstraction_path(), [pol], note="test fixture")
        st = os.path.join(d, "state.bin")
        if os.path.exists(st):
            os.remove(st)
    return man, game, solve


def solver_for(game, solve):
    """A fresh solver on the same tree (for walking node ids)."""
    return Solver.nlhe(game, T.abstraction_path(), Solve(algo=solve.algo, seed=solve.seed, precision="float",
                                                          avg_mode="off"))
