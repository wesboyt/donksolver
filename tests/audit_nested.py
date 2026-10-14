"""The nested abstraction computes what it claims.

Built with 6/6/8 equity bins, 4 potential sub-bins and board classes on every
street (flop: suits; turn, river: texture).

  1. e1 and e2 of sampled holdings == brute force over every runout with the
     one-holding hand-strength routine (turn: 46 rivers; flop: 47 turns x 46
     rivers)
  2. the stored bucket == the nested rule re-derived here from e1/e2: equal
     width equity bins between the board's extremes, then equal-width e2
     sub-bins inside each bin
     controls: equal-MASS equity bins, and sub-bins cut by equity instead of
     e2, must both disagree with the table
  3. the full bucket == board class * span + stored bucket, with the board
     class re-derived here; river == class * bins + equal-width HS bin
     control: a board class that ignores suits must disagree
  4. buckets are invariant under suit permutations of (hole, board)
     control: permuting the hole's suits only must change some buckets
  5. potential separates draws: on two-tone flops, among holdings in the same
     equity bin, four-flush holdings sit in a higher sub-bin on average than
     holdings with no flush draw
"""
import ctypes as C
import itertools
import os
import random

import numpy as np

import _common as T
from donksolver import Abstraction
from donksolver import _native as N

BINS, SUB, BOARDS = (6, 6, 8), 4, ("suits", "texture", "texture")
MODE = {"none": 0, "suits": 1, "texture": 2}


def path():
    p = os.path.join(T.ARTIFACTS, "nested_6_6_8.dab")
    if not os.path.exists(p):
        os.makedirs(T.ARTIFACTS, exist_ok=True)
        print("  [fixture] building %s" % p, flush=True)
        ab = Abstraction.build(BINS, threads=T.THREADS, kind="nested", sub=SUB, boards=BOARDS)
        tmp = "%s.%d.tmp" % (p, os.getpid())
        ab.save(tmp)
        os.replace(tmp, p)
    return p


def hs(hole, board5):
    return N.lib.ds_hand_strength((C.c_int32 * 2)(*hole), (C.c_int32 * 5)(*board5))


def hidx(a, b):
    a, b = min(a, b), max(a, b)
    return b * (b - 1) // 2 + a


def metrics(ab, board):
    e1 = np.zeros(1326)
    e2 = np.zeros(1326)
    tab = np.zeros(1326, dtype=np.int32)
    bi = N.lib.ds_abs_debug_nested(ab._h, len(board), (C.c_int32 * 5)(*(board + [0] * (5 - len(board)))),
                                   N.ptr(e1, C.c_double), N.ptr(e2, C.c_double), N.ptr(tab, C.c_int32))
    N.check(bi >= 0)
    return e1, e2, tab


def width_bin(v, B, lo, hi):
    w = hi - lo + 1e-8
    for k in range(1, B + 1):
        if v <= lo + w * (k / B):
            return k - 1
    return B - 1


def nested(e1, e2, n_bins, sub, cut_by_equity=False, equal_mass=False):
    ok = np.where(e1 >= 0)[0]
    out = np.full(1326, -1)
    if equal_mass:
        order = ok[np.argsort(e1[ok], kind="stable")]
        b1 = {h: min(n_bins - 1, i * n_bins // len(order)) for i, h in enumerate(order)}
    else:
        lo, hi = e1[ok].min(), e1[ok].max()
        b1 = {h: width_bin(e1[h], n_bins, lo, hi) for h in ok}
    second = e1 if cut_by_equity else e2
    for k in range(n_bins):
        members = [h for h in ok if b1[h] == k]
        if not members:
            continue
        lo, hi = second[members].min(), second[members].max()
        for h in members:
            out[h] = k * sub + width_bin(second[h], sub, lo, hi)
    return out


def suit_class(board):
    c = sorted(np.bincount([x & 3 for x in board], minlength=4), reverse=True)
    if len(board) == 3:
        return c[0] - 1
    if len(board) == 4:
        return 4 if c[0] == 4 else 3 if c[0] == 3 else 2 if c[1] == 2 else 1 if c[0] == 2 else 0
    return 3 if c[0] >= 5 else 2 if c[0] == 4 else 1 if c[0] == 3 else 0


def board_class(mode, board, ignore_suits=False):
    if mode == "none":
        return 0
    s = 0 if ignore_suits else suit_class(board)
    if mode == "suits":
        return s
    a = sum((x >> 2) >= 5 for x in board)
    b = sum((x >> 2) >= 9 for x in board)
    n = len(board)
    return s * ((n + 1) * (n + 2) // 2) + a * (a + 1) // 2 + b


def main():
    ab = Abstraction.load(path())
    nb = ab.buckets
    span = [169, BINS[0] * SUB, BINS[1] * SUB, BINS[2]]
    rng = random.Random(31)

    # 1. metrics vs brute force
    worst = 0.0
    for _ in range(12):
        cards = rng.sample(range(52), 6)
        board, hole = cards[:4], cards[4:]
        e1, e2, _ = metrics(ab, board)
        v = [hs(hole, board + [r]) for r in range(52) if r not in cards]
        h = hidx(*hole)
        worst = max(worst, abs(e1[h] - np.mean(v)), abs(e2[h] - np.mean(np.square(v))))
    T.check("turn e1, e2 == brute force over the rivers", worst < 1e-12, "12 spots, worst diff=%.2e" % worst)
    worst = 0.0
    for _ in range(4):
        cards = rng.sample(range(52), 5)
        board, hole = cards[:3], cards[3:]
        e1, e2, _ = metrics(ab, board)
        live = [c for c in range(52) if c not in cards]
        et = [np.mean([hs(hole, board + [t, r]) for r in live if r != t]) for t in live]
        h = hidx(*hole)
        worst = max(worst, abs(e1[h] - np.mean(et)), abs(e2[h] - np.mean(np.square(et))))
    T.check("flop e1, e2 == brute force over the turns and rivers", worst < 1e-12, "4 spots, worst diff=%.2e" % worst)

    # 2. the nested rule, 3. board classes
    bad = bad_mass = bad_cut = bad_full = bad_nosuit = rows = 0
    draws = [[], []]
    for n_board, street in ((3, 1), (4, 2)):
        for _ in range(25):
            board = rng.sample(range(52), n_board)
            e1, e2, tab = metrics(ab, board)
            want = nested(e1, e2, BINS[street - 1], SUB)
            bad += int(np.sum(want != tab))
            bad_mass += int(np.sum(nested(e1, e2, BINS[street - 1], SUB, equal_mass=True) != tab))
            bad_cut += int(np.sum(nested(e1, e2, BINS[street - 1], SUB, cut_by_equity=True) != tab))
            cls = board_class(BOARDS[street - 1], board)
            flat = board_class(BOARDS[street - 1], board, ignore_suits=True)
            for a, b in rng.sample(list(itertools.combinations([c for c in range(52) if c not in board], 2)), 40):
                got = ab.bucket(street, [a, b], board)
                rows += 1
                bad_full += got != cls * span[street] + tab[hidx(a, b)] or got >= nb[street]
                bad_nosuit += got != flat * span[street] + tab[hidx(a, b)]
            if n_board == 3 and suit_class(board) == 1:
                two = max(range(4), key=lambda s: sum((c & 3) == s for c in board))
                for a, b in itertools.combinations([c for c in range(52) if c not in board], 2):
                    k = tab[hidx(a, b)]
                    flush_draw = (a & 3) == two and (b & 3) == two
                    no_draw = (a & 3) != two and (b & 3) != two
                    if flush_draw or no_draw:
                        draws[0 if no_draw else 1].append((k // SUB, k % SUB))
    T.check("stored bucket == nested rule (equity bin, then potential sub-bin)", bad == 0,
            "50 boards, mismatches=%d" % bad)
    T.control("equal-mass equity bins detected", bad_mass > 0, "holdings differing=%d" % bad_mass)
    T.control("sub-bins cut by equity instead of potential detected", bad_cut > 0, "holdings differing=%d" % bad_cut)
    T.check("bucket == board class * span + stored bucket (flop, turn)", bad_full == 0,
            "rows=%d mismatches=%d" % (rows, bad_full))
    T.control("board class that ignores suits detected", bad_nosuit > 0, "rows differing=%d" % bad_nosuit)

    bad = ctl = 0
    for _ in range(600):
        cards = rng.sample(range(52), 7)
        hole, board = cards[:2], cards[2:]
        got = ab.bucket(3, hole, board)
        bad += got != board_class(BOARDS[2], board) * BINS[2] + width_bin(hs(hole, board), BINS[2], 0.0, 1.0 - 1e-8)
        ctl += got != board_class(BOARDS[2], board, ignore_suits=True) * BINS[2] + width_bin(
            hs(hole, board), BINS[2], 0.0, 1.0 - 1e-8)
    T.check("river bucket == board class * bins + equal-width HS bin", bad == 0, "600 spots, mismatches=%d" % bad)
    T.control("river board class that ignores suits detected", ctl > 0, "spots differing=%d" % ctl)

    # 4. suit isomorphism
    for street, nbd in ((1, 3), (2, 4), (3, 5)):
        diff = ctl = 0
        trials = 1500 if street < 3 else 400
        for _ in range(trials):
            cards = rng.sample(range(52), 2 + nbd)
            perm = list(range(4))
            rng.shuffle(perm)
            mp = lambda c: (c >> 2) * 4 + perm[c & 3]
            base = ab.bucket(street, cards[:2], cards[2:])
            diff += ab.bucket(street, [mp(c) for c in cards[:2]], [mp(c) for c in cards[2:]]) != base
            ctl += ab.bucket(street, [mp(c) for c in cards[:2]], cards[2:]) != base
        T.check("street %d buckets invariant under suit permutation" % street, diff == 0,
                "trials=%d changed=%d" % (trials, diff))
        T.control("street %d hole-only permutation detected" % street, ctl > 0, "changed=%d" % ctl)

    # 5. potential separates flush draws inside an equity bin
    gap = []
    for k in range(BINS[0]):
        none = [s for b, s in draws[0] if b == k]
        fd = [s for b, s in draws[1] if b == k]
        if len(none) >= 50 and len(fd) >= 50:
            gap.append(np.mean(fd) - np.mean(none))
    T.check("flush draws sit in higher potential sub-bins than no-draw holdings of the same equity bin",
            len(gap) >= 2 and np.mean(gap) > 0.3,
            "equity bins compared=%d mean sub-bin gap=%.2f (per bin %s)" % (len(gap), np.mean(gap) if gap else 0,
                                                                         ["%.2f" % g for g in gap]))


if __name__ == "__main__":
    main()
    T.finish()
