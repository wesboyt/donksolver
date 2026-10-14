"""The chance-sampled regret increment is unbiased for the counterfactual regret.

Leduc (2 and 3 players) after a short solve, so sigma is far from uniform. The
deals are enumerable, so for every table entry both the exact mean E[inc] and
the exact second moment E[inc^2] are computed; the mean of N sampled deals is
then tested with its TRUE standard error sqrt(Var/N) -- no variance estimate,
so rare-deal entries cannot inflate the statistic. Pass if sum z^2 matches its
degrees of freedom and no |z| is extreme (Bonferroni over entries).
control: dropping the opponent reach factor q must be rejected by the same test.
"""
import math

import numpy as np

import _common as T
from donksolver import Solver, toy


def test(exact, second, sampled, n):
    var = np.maximum(second - exact * exact, 0.0)
    keep = var > 1e-18
    z = (sampled[keep] - exact[keep]) / np.sqrt(var[keep] / n)
    zero_bad = int(np.sum(np.abs(sampled[~keep] - exact[~keep]) > 1e-9))
    df = int(np.sum(keep))
    chi = float(np.sum(z * z))
    zc = ((chi / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return float(np.max(np.abs(z))), chi / df, zc, df, zero_bad


def main():
    n = 400000
    for players, warm in ((2, 3000), (3, 1500)):
        s = Solver(toy("leduc", players, algo="cs", avg_mode="off", seed=9))
        s.run(warm, 1)
        for player in range(players):
            exact = s.debug_regret(player, exact=True)
            s2 = np.zeros_like(exact)
            from donksolver import _native as N
            import ctypes as C
            N.check(N.lib.ds_solver_debug_regret(s._h, player, 2, 0, 0, 0, N.ptr(s2, C.c_double)))
            sampled = s.debug_regret(player, exact=False, samples=n, seed=1000 + player)
            zmax, ratio, zc, df, zb = test(exact, s2, sampled, n)
            bonf = math.sqrt(2) * math.sqrt(math.log(df / 0.001 * 2 / math.pi))  # ~ normal quantile at 0.001/df
            T.check("leduc%d seat %d: sampled regret unbiased" % (players, player),
                    zc < 4 and zmax < max(bonf, 4.5) and zb == 0,
                    "entries=%d chi2/df=%.3f z_chi=%.2f max|z|=%.2f (bound %.2f)" % (df, ratio, zc, zmax, bonf))
            broken = s.debug_regret(player, exact=True, broken=True)
            zmax_b, ratio_b, zc_b, _, _ = test(broken, s2, sampled, n)
            T.control("leduc%d seat %d: missing opponent reach rejected" % (players, player), zc_b > 10,
                      "chi2/df=%.1f max|z|=%.1f" % (ratio_b, zmax_b))


if __name__ == "__main__":
    main()
    T.finish()
