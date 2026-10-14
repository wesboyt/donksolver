"""The range deal sampler draws hole cards with law prod_i min(w_i,1) * 1[disjoint] / Z0.

Three seats with overlapping small ranges and non-trivial weights (some >= 1),
a dead board card; the enumerated joint law is compared with 400k draws by a
chi-square test (Wilson-Hilferty z). control: redrawing only the colliding seat
(a common implementation shortcut) must be rejected.
"""
import ctypes as C
import itertools
import math
import random

import numpy as np

import _common as T
from donksolver import _native as N


def chi2_z(counts, probs, n):
    exp = probs * n
    keep = exp > 0
    stat = float(np.sum((counts[keep] - exp[keep]) ** 2 / exp[keep]))
    df = int(np.sum(keep)) - 1
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return stat, df, z, float(np.sum(counts[~keep]))


def main():
    rng = random.Random(3)
    cards = list(range(9))
    seats = [
        [(a, b) for a, b in itertools.combinations([0, 1, 2, 3, 4, 5], 2)],
        [(a, b) for a, b in itertools.combinations([2, 3, 4, 5, 6, 7], 2)],
        [(a, b) for a, b in itertools.combinations([0, 3, 5, 7, 8], 2)],
    ]
    weights = [[round(rng.uniform(0.05, 1.6), 3) for _ in s] for s in seats]
    board = [8]
    # enumerate the law
    tuples = list(itertools.product(*[range(len(s)) for s in seats]))
    index = {t: i for i, t in enumerate(tuples)}
    p = np.zeros(len(tuples))
    for t, i in index.items():
        used = set(board)
        ok = True
        w = 1.0
        for s, k in enumerate(t):
            c = seats[s][k]
            if c[0] in used or c[1] in used:
                ok = False
                break
            used.update(c)
            w *= min(weights[s][k], 1.0)
        p[i] = w if ok else 0.0
    p /= p.sum()

    nc = (C.c_int32 * 3)(*[len(s) for s in seats])
    flat = [c for s in seats for pair in s for c in pair]
    combos = (C.c_int32 * len(flat))(*flat)
    wflat = [w for ws in weights for w in ws]
    wts = (C.c_double * len(wflat))(*wflat)
    bd = (C.c_int32 * 1)(*board)
    n = 400000
    for broken in (0, 1):
        out = np.zeros(n * 3, dtype=np.int32)
        N.check(N.lib.ds_sample_ranges(3, nc, combos, wts, bd, 1, n, 17 + broken, broken, N.ptr(out, C.c_int32)))
        out = out.reshape(n, 3)
        cnt = np.zeros(len(tuples))
        for row in map(tuple, out):
            cnt[index[row]] += 1
        stat, df, z, impossible = chi2_z(cnt, p, n)
        detail = "chi2=%.0f df=%d z=%.2f impossible=%d" % (stat, df, z, impossible)
        if broken:
            T.control("seat-only redraw rejected", z > 6 or impossible > 0, detail)
        else:
            T.check("sampler law == enumerated law", z < 4 and impossible == 0, detail)


if __name__ == "__main__":
    main()
    T.finish()
