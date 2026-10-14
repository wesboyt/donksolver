"""The Bernoulli(p) average pass converges to the reach-weighted linear average.

Leduc 2p, chance sampling, p_avg = 0.25, one thread. Alongside the real
table the solver keeps a SHADOW that adds the exact expected increment
(all deals enumerated) every iteration. The mass-weighted mean absolute
difference of the normalised strategies must be small and must shrink as T
grows. control: an average pass that also weights by the opponents' action
probabilities (the naive external-sampling average) must stay far off.
"""
import numpy as np

import _common as T
from donksolver import Solver, toy


def weighted_err(s):
    # worst over rows with material mass, and the mass-weighted mean
    worst, _ = s.shadow_compare(min_mass=1e3)
    return worst


def run(broken, sched):
    s = Solver(toy("leduc", 2, algo="cs", avg_mode="weighted", p_avg=0.25, seed=5))
    s.shadow(True, broken)
    out, done = [], 0
    for t in sched:
        s.run(t - done, 1)
        done = t
        out.append(weighted_err(s))
    return out


def main():
    sched = (4000, 16000, 64000)
    ok = run(False, sched)
    bad = run(True, sched)
    T.check("sampled average tracks the exact linear average", ok[-1] < 0.1,
            "worst|mass>=1e3 at T=%s: %s" % (sched, ["%.3f" % x for x in ok]))
    T.check("and the error shrinks with T", ok[-1] < 0.6 * ok[0], "")
    T.control("opponent-weighted average rejected", bad[-1] > 0.3 and bad[-1] > 4 * ok[-1],
              "worst at T=%s: %s" % (sched, ["%.3f" % x for x in bad]))


if __name__ == "__main__":
    main()
    T.finish()
