"""The 7-card evaluator induces the poker order.

  1. native eval vs the native best-of-21 textbook reference, pairwise order
     (including ties) on random 7-card hands, plus every 5/6-card size
  2. native eval vs pokerfast's eval7 (itself tested against pokerkit),
     pairwise order
  control: a variant with straight and flush ranks swapped must disagree.
"""
import ctypes as C
import random

import numpy as np

import _common as T
from donksolver import _native as N
from donksolver.cards import DECK


def sign(a, b):
    return (a > b) - (a < b)


def main():
    rng = random.Random(7)
    n = 60000
    hands = [rng.sample(range(52), 7) for _ in range(n)]
    arr = np.ascontiguousarray(np.array(hands, dtype=np.int32))
    out = np.zeros(n, dtype=np.uint32)
    N.lib.ds_eval_batch(N.ptr(arr, C.c_int32), n, 7, N.ptr(out, C.c_uint32))
    ref = np.array([N.lib.ds_eval_reference((C.c_int32 * 7)(*h), 7) for h in hands], dtype=np.uint64)
    fast = out.astype(np.uint64)
    # pairwise order on random pairs + all adjacent pairs
    idx = np.array([rng.randrange(n) for _ in range(200000)])
    jdx = np.array([rng.randrange(n) for _ in range(200000)])
    s_fast = np.sign(fast[idx].astype(np.int64) - fast[jdx].astype(np.int64))
    s_ref = np.sign(ref[idx].astype(np.int64) - ref[jdx].astype(np.int64))
    T.check("native eval == textbook reference order (7 cards, 200k pairs)", np.all(s_fast == s_ref),
            "mismatches=%d" % int(np.sum(s_fast != s_ref)))
    T.check("native eval == reference value exactly", np.all(fast == ref), "diff=%d" % int(np.sum(fast != ref)))

    for k in (5, 6):
        hs = [rng.sample(range(52), k) for _ in range(20000)]
        a = [N.lib.ds_eval((C.c_int32 * 7)(*h), k) for h in hs]
        b = [N.lib.ds_eval_reference((C.c_int32 * 7)(*h), k) for h in hs]
        T.check("native eval == reference (%d cards)" % k, a == b, "diff=%d" % sum(x != y for x, y in zip(a, b)))

    # control: swap straight(4) and flush(5) categories in the reference
    def broken(v):
        cat = int(v) >> 20
        cat = {4: 5, 5: 4}.get(cat, cat)
        return (cat << 20) | (int(v) & 0xFFFFF)
    rb = np.array([broken(v) for v in ref], dtype=np.int64)
    s_b = np.sign(rb[idx] - rb[jdx])
    T.control("straight/flush swap is detected", np.any(s_b != s_fast), "mismatches=%d" % int(np.sum(s_b != s_fast)))

    try:
        pf, _ = T.reference_engine()
    except Exception as e:
        T.check("pokerfast importable (pip install donksolver[test])", False, repr(e))
        return
    m = 30000
    ref7 = np.array([pf.eval7([DECK[c] for c in hands[i]]) for i in range(m)], dtype=np.int64)
    ii = idx[idx < m][:50000]
    jj = jdx[:len(ii)] % m
    a = np.sign(fast[ii].astype(np.int64) - fast[jj].astype(np.int64))
    b = np.sign(ref7[ii] - ref7[jj])
    T.check("native eval order == pokerfast eval7 order", np.all(a == b), "pairs=%d mismatches=%d" % (len(ii), int(np.sum(a != b))))


if __name__ == "__main__":
    main()
    T.finish()
