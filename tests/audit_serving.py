"""Serving: what the caller receives is what the table says.

6-max-lite fixture (40bb). Real states come from pokerfast's FastHand engine.

  1. frequencies: first-to-act (UTG) rows over all 169 preflop classes, random
     stacks near the class depth; realised action counts vs the table's
     probabilities, pooled chi-square. control: a row-permuted table is rejected
  2. whole hands through donksolver.engine.Player, every seat either the
     solver or a random-legal player: no exceptions, every action legal, and
     the status rates are reported
  3. throughput of the batch server (rows/s at batch 64 and 512)
"""
import math
import random
import time

import numpy as np

import _common as T
from fixtures import policy
from donksolver import Bundle
from donksolver.engine import Player

KIND_TO_ROOT = {0: "fold", 2: "call", 3: "raise", 4: "allin"}


def chi2_z(stat, df):
    return ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))


def main():
    man, game, _ = policy("6max-lite", 40, 300000)
    b = Bundle(man)
    rng = random.Random(9)
    n = 6
    bb = game.bb

    # 1. root frequencies
    R = 120000
    stacks = [[bb * rng.randint(36, 44) for _ in range(n)] for _ in range(R)]
    holes = [rng.sample(range(52), 2) for _ in range(R)]
    u = [rng.random() for _ in range(R)]
    out = b.act(stacks, [[game.sb, bb]] * R, holes, [[] for _ in range(R)], [[] for _ in range(R)], u)
    T.check("root rows all served", np.all(out["status"] == 0), "statuses=%s" % np.bincount(out["status"]))
    node = int(out["node"][0])
    probs, kinds, _ = b.strategy(0, node, 0)
    kind_index = {k: i for i, k in enumerate(kinds)}  # abstract kinds: 0 fold 1 call 2 raise 3 allin
    real_to_abs = {0: 0, 2: 1, 3: 2, 4: 3, 1: 1}
    counts = np.zeros((169, len(kinds)))
    for r in range(R):
        counts[out["bucket"][r], kind_index[real_to_abs[int(out["kind"][r])]]] += 1
    table = np.array([b.strategy(0, node, bk)[0] for bk in range(169)], dtype=np.float64)
    table /= table.sum(axis=1, keepdims=True)

    def pooled(tab):
        stat, df = 0.0, 0
        for bk in range(169):
            nb = counts[bk].sum()
            exp = tab[bk] * nb
            keep = exp > 5
            if keep.sum() < 2:
                continue
            stat += float(np.sum((counts[bk][keep] - exp[keep]) ** 2 / exp[keep]))
            df += int(keep.sum()) - 1
        return stat, df

    stat, df = pooled(table)
    z = chi2_z(stat, df)
    T.check("served frequencies == table (UTG, 169 classes)", z < 4, "chi2=%.1f df=%d z=%.2f" % (stat, df, z))
    perm = table[np.random.RandomState(1).permutation(169)]
    stat_b, df_b = pooled(perm)
    T.control("row-permuted table rejected", chi2_z(stat_b, df_b) > 10, "z=%.1f" % chi2_z(stat_b, df_b))

    # 2. whole hands with mixed seats
    fast_hand, _ = T.reference_engine()
    opp = Player(man, log=None)
    illegal = errors = 0
    hands = 1500
    t_act = 0.0
    rows = 0
    for _ in range(hands):
        h = fast_hand.FastHand(rng=random.Random(rng.randrange(1 << 30)))
        solver_seats = {s for s in range(n) if rng.random() < 0.6}
        while not h.done:
            sp = h.get_action_space()
            seat = sp["player"]
            if seat in solver_seats:
                t0 = time.perf_counter()
                acts, sizes, _, status = opp.act([h])
                t_act += time.perf_counter() - t0
                rows += 1
                a, c = acts[0], sizes[0]
                if status[0] >= 2:
                    a, c = ("call", sp["call"]) if "call" in sp else ("check", 0)
            else:
                opts = [("check", 0)] if "check" in sp else [("fold", 0), ("call", sp["call"])]
                if "min_bet" in sp:
                    opts.append(("raise", rng.randint(sp["min_bet"], sp["max_bet"])))
                    opts.append(("allin", sp["max_bet"]))
                a, c = rng.choice(opts)
            legal = ((a == "fold" and "call" in sp) or (a == "check" and "check" in sp)
                     or (a == "call" and "call" in sp and c == sp["call"])
                     or (a == "raise" and "min_bet" in sp and sp["min_bet"] <= c <= sp["max_bet"])
                     or (a == "allin" and (("min_bet" in sp and c == sp["max_bet"]) or
                                           ("min_bet" not in sp and "call" in sp))))
            illegal += not legal
            try:
                if a == "fold":
                    h.fold()
                elif a == "check":
                    h.check()
                elif a == "call":
                    h.call()
                elif a == "raise":
                    h.bet_or_raise(c)
                else:
                    h.bet_or_raise(sp["max_bet"]) if "max_bet" in sp else h.call()
            except Exception as e:  # pragma: no cover
                errors += 1
                break
    T.check("mixed-seat hands: no errors, all solver actions legal", illegal == 0 and errors == 0,
            "hands=%d solver rows=%d illegal=%d errors=%d" % (hands, rows, illegal, errors))
    print("  status/kind rates:", opp.rates())
    print("  single-row adapter latency: %.0f us/row (includes FastHand packing)" % (1e6 * t_act / max(rows, 1)))

    # 3. batch throughput of the native server
    for B in (64, 512):
        hands_b = []
        while len(hands_b) < B:
            h = fast_hand.FastHand(rng=random.Random(rng.randrange(1 << 30)))
            for _ in range(rng.randint(0, 5)):
                if h.done:
                    break
                sp = h.get_action_space()
                h.call() if "call" in sp else h.check()
            if not h.done:
                hands_b.append(h)
        t0 = time.perf_counter()
        reps = 20
        for _ in range(reps):
            opp.act(hands_b)
        dt = time.perf_counter() - t0
        print("  batch %d: %.0f rows/s (%.1f us/row incl. packing)" % (B, B * reps / dt, 1e6 * dt / (B * reps)))


if __name__ == "__main__":
    main()
    T.finish()
