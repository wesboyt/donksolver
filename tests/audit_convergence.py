"""Convergence on the validation games (exact best responses, all deals enumerated).

  1. Kuhn 2p: the average profile's value for seat 0 -> -1/18 and NashConv -> 0.
     4e6 iterations: over 16 runs (8 seeds, 1 and 2 threads) NashConv was at
     most 0.0012 there, against 0.0054 at 2e5, which the bound of 0.005 cannot
     separate from a failure.
  2. Leduc 2p, LCFR (es): NashConv falls at least like 1/sqrt(T) (log-log slope
     <= -0.4 per decade over 1e4 -> 1e6, median of 3 seeds).
  3. linear_avg weighting (uniform regrets, linear average) is not better than
     LCFR at 1e6 iterations (median of 3 seeds); both ratios are reported.
  4. Leduc 3p: reports each seat's best-response gain. With three players CFR
     converges to a coarse correlated equilibrium at best, so the only check is
     that the gain falls.
  control: serving the CURRENT strategy (no average) does not converge -- the
  same slope test must fail on it.
"""
import numpy as np

import _common as T
from donksolver import Solver, toy


def curve(game, players, weighting, avg, seed, sched, threads):
    s = Solver(toy(game, players, algo="es", weighting=weighting, avg_mode=avg, p_avg=1.0, seed=seed))
    done, out = 0, []
    for t in sched:
        s.run(t - done, threads)
        done = t
        x = s.exploitability()
        out.append(x)
    return out


def main():
    th = T.THREADS
    k = curve("kuhn", 2, "linear", "weighted", 1, (4000000,), th)[-1]
    T.check("kuhn value -> -1/18", abs(k["ev"][0] + 1 / 18) < 2e-3, "ev0=%.5f" % k["ev"][0])
    T.check("kuhn nashconv small", k["nashconv"] < 5e-3, "nashconv=%.5f" % k["nashconv"])

    sched = (10**4, 10**5, 10**6)
    lin, mk, cur = [], [], []
    for seed in (1, 2, 3):
        lin.append([x["nashconv"] for x in curve("leduc", 2, "linear", "weighted", seed, sched, th)])
        mk.append([x["nashconv"] for x in curve("leduc", 2, "linear_avg", "weighted", seed, sched, th)])
        cur.append([x["nashconv"] for x in curve("leduc", 2, "linear", "off", seed, sched, th)])
    lin, mk, cur = np.array(lin), np.array(mk), np.array(cur)
    slope = np.median(np.log10(lin[:, 2] / lin[:, 0]) / 2)
    T.check("leduc2 LCFR nashconv slope <= -0.4/decade", slope <= -0.4,
            "median slope=%.2f nashconv@1e6=%s" % (slope, ["%.4f" % x for x in lin[:, 2]]))
    ratio = np.median(mk[:, 2] / lin[:, 2])
    T.check("linear_avg weighting not better than LCFR at 1e6", ratio >= 1.0,
            "median nashconv ratio linear_avg/linear=%.2f (per seed %s)" % (
                ratio, ["%.2f" % r for r in mk[:, 2] / lin[:, 2]]))
    cslope = np.median(np.log10(cur[:, 2] / cur[:, 0]) / 2)
    T.control("current strategy (no average) fails the slope test", cslope > -0.4,
              "median slope=%.2f nashconv@1e6=%s" % (cslope, ["%.3f" % x for x in cur[:, 2]]))

    three = curve("leduc", 3, "linear", "weighted", 1, (3 * 10**4, 3 * 10**5), th)
    g0, g1 = three[0]["gains"], three[1]["gains"]
    T.check("leduc3 best-response gains fall (CCE sense only)", max(g1) < max(g0),
            "max gain %.4f -> %.4f; per seat %s" % (max(g0), max(g1), ["%.4f" % g for g in g1]))


if __name__ == "__main__":
    main()
    T.finish()
