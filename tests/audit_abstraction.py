"""The card abstraction computes what it claims.

  1. river HS (native, one pass) == brute-force count over all 990 opponent
     holdings with the native evaluator, exactly
  2. buckets are invariant under suit permutations of (hole, board) on every
     street -- the isomorphism is lossless
     control: permuting the HOLE's suits only must change some buckets
  3. percentile buckets carry roughly equal mass under uniform deals
  4. flop and turn buckets predict final hand strength: the mean river HS by
     bucket is increasing (Spearman rho > 0.9)
"""
import ctypes as C
import random

import numpy as np

import _common as T
from donksolver import Abstraction
from donksolver import _native as N


def hs_brute(hole, board):
    dead = set(hole) | set(board)
    live = [c for c in range(52) if c not in dead]
    me = N.lib.ds_eval((C.c_int32 * 7)(*(hole + board)), 7)
    opp = [(a, b) for i, a in enumerate(live) for b in live[i + 1:]]
    arr = np.array([[a, b] + board for a, b in opp], dtype=np.int32)
    out = np.zeros(len(opp), dtype=np.uint32)
    N.lib.ds_eval_batch(N.ptr(np.ascontiguousarray(arr), C.c_int32), len(opp), 7, N.ptr(out, C.c_uint32))
    return (np.sum(out < me) + 0.5 * np.sum(out == me)) / len(opp)


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ab = Abstraction.load(T.abstraction_path())
    B = ab.buckets
    rng = random.Random(21)

    bad = 0
    for _ in range(300):
        cards = rng.sample(range(52), 7)
        h, b = cards[:2], cards[2:]
        got = N.lib.ds_hand_strength((C.c_int32 * 2)(*h), (C.c_int32 * 5)(*b))
        bad += abs(got - hs_brute(h, b)) > 1e-12
    T.check("river HS == brute force (300 spots)", bad == 0, "mismatches=%d" % bad)

    for street, nb in ((1, 3), (2, 4), (3, 5)):
        diff = ctl = 0
        trials = 1500 if street < 3 else 400
        for _ in range(trials):
            cards = rng.sample(range(52), 2 + nb)
            perm = list(range(4))
            rng.shuffle(perm)
            mp = lambda c: (c >> 2) * 4 + perm[c & 3]
            base = ab.bucket(street, cards[:2], cards[2:])
            diff += ab.bucket(street, [mp(c) for c in cards[:2]], [mp(c) for c in cards[2:]]) != base
            ctl += ab.bucket(street, [mp(c) for c in cards[:2]], cards[2:]) != base
        T.check("street %d buckets invariant under suit permutation" % street, diff == 0,
                "trials=%d changed=%d" % (trials, diff))
        T.control("street %d hole-only permutation detected" % street, ctl > 0, "changed=%d" % ctl)

    for street, nb in ((1, 3), (2, 4)):
        cnt = np.zeros(B[street])
        final = [[] for _ in range(B[street])]
        n = 30000
        for _ in range(n):
            cards = rng.sample(range(52), 7)
            bk = ab.bucket(street, cards[:2], cards[2:2 + nb])
            cnt[bk] += 1
            if len(final[bk]) < 60:
                final[bk].append(N.lib.ds_hand_strength((C.c_int32 * 2)(*cards[:2]), (C.c_int32 * 5)(*cards[2:])))
        share = cnt / n
        T.check("street %d bucket mass roughly equal" % street,
                share.max() < 2.0 / B[street] and share.min() > 0.4 / B[street],
                "min=%.4f max=%.4f target=%.4f" % (share.min(), share.max(), 1 / B[street]))
        means = np.array([np.mean(f) if f else np.nan for f in final])
        ok = ~np.isnan(means)
        rho = spearman(np.arange(B[street])[ok], means[ok])
        T.check("street %d bucket predicts river HS" % street, rho > 0.9, "spearman=%.3f" % rho)


if __name__ == "__main__":
    main()
    T.finish()
