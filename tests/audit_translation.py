"""Action translation and real-state conversion.

Heads-up fixture (20bb, one size per street + all-in), equal stacks so the
abstract tree and the real game coincide on-tree.

  1. on-tree round trip: random abstract lines, replayed as real histories,
     land on the same abstract node with status ok and no translation
  2. pseudo-harmonic mapping: a real raise of x = 3 pot between abstract sizes
     A and B maps to A with probability (B-x)(1+A)/((B-A)(1+x)), within SE over
     many independent hands
     control: the linear mapping (B-x)/(B-A) is rejected
  3. every served action is legal in the real engine and executes as labelled
     (random stacks, whole hands, both seats served; pokerfast's FastHand)
"""
import math
import random

import numpy as np

import _common as T
from fixtures import policy, solver_for
from donksolver import Bundle
from donksolver.engine import Player


def main():
    man, game, solve = policy("tiny", 20, 400000)
    b = Bundle(man)
    s = solver_for(game, solve)
    rng = random.Random(5)
    stack = int(game.stack_bb * game.bb)

    # 1. on-tree round trip
    bad = checked = 0
    for _ in range(3000):
        node, hist = 0, []
        depth = rng.randrange(0, 6)
        for _ in range(depth):
            nd = s.node(node)
            if nd["actor"] < 0:
                break
            a = rng.randrange(nd["nA"])
            hist.append((nd["actor"], 1 if nd["kinds"][a] == 0 else 0, nd["chips"][a]))
            node = nd["children"][a]
        nd = s.node(node)
        if nd["actor"] < 0:
            continue
        street = nd["street"]
        cards = rng.sample(range(52), 7)
        board = cards[2:2 + (0 if street == 0 else street + 2)]
        out = b.act([[stack, stack]], [[game.sb, game.bb]], [cards[:2]], [board], [hist], [0.5])
        checked += 1
        if not (out["status"][0] == 0 and out["node"][0] == node and out["offtree"][0] == 0):
            bad += 1
            if bad <= 3:
                print("  mismatch", hist, node, {k: v[0] for k, v in out.items()})
    T.check("on-tree histories land on the same node", bad == 0 and checked > 1000,
            "checked=%d mismatches=%d" % (checked, bad))

    # 2. pseudo-harmonic mapping at the root
    root = s.node(0)
    rz = [i for i, k in enumerate(root["kinds"]) if k >= 2]
    A, Bf = root["fracs"][rz[0]], root["fracs"][rz[1]]
    x = 3.0
    pot, coc, mb, bet0 = 300, 100, 200, 100
    chips = int(round(mb + x * (pot + coc) - bet0))
    xr = (bet0 + chips - mb) / (pot + coc)
    p_h = (Bf - xr) * (1 + A) / ((Bf - A) * (1 + xr))
    p_lin = (Bf - xr) / (Bf - A)
    # The mapping coin is a hash of what the acting seat knows, so rows must
    # differ in that information to be independent draws: every row gets its
    # own stack depth (which does not move the real pot fraction x).
    n = 6000
    holes = [rng.sample(range(52), 2) for _ in range(n)]
    stacks = [[stack + 7 * i, stack + 11 * i] for i in range(n)]
    out = b.act(stacks, [[game.sb, game.bb]] * n, holes, [[] for _ in range(n)],
                [[(0, 0, chips)]] * n, [0.5] * n)
    child_A = root["children"][rz[0]]
    ok = out["status"] < 2
    # node reached is the child of A (or of B): read its parent choice from the node id
    nodes_A = set()
    nA = s.node(child_A)
    nodes_A.add(child_A)
    hit = np.mean(out["node"][ok] == child_A)
    se = math.sqrt(p_h * (1 - p_h) / ok.sum())
    T.check("pseudo-harmonic mapping frequency", abs(hit - p_h) < 4 * se,
            "A=%.3f B=%.3f x=%.3f observed=%.4f expected=%.4f se=%.4f" % (A, Bf, xr, hit, p_h, se))
    T.control("linear mapping rejected", abs(hit - p_lin) > 6 * se, "linear=%.4f" % p_lin)

    # 3. legality in the real engine, whole hands, random stacks
    fast_hand, _ = T.reference_engine()
    opp = Player(man, log=None)
    illegal = 0
    rows = 0
    hands = 1500
    for _ in range(hands):
        h = fast_hand.FastHand(player_count=2, rng=random.Random(rng.randrange(1 << 30)))
        while not h.done:
            sp = h.get_action_space()
            acts, sizes, probs, status = opp.act([h])
            a, c = acts[0], sizes[0]
            rows += 1
            if status[0] >= 2:
                h.call() if "call" in sp else h.check()
                continue
            legal = ((a == "fold" and "call" in sp) or (a == "check" and "check" in sp)
                     or (a == "call" and "call" in sp and c == sp["call"])
                     or (a == "raise" and "min_bet" in sp and sp["min_bet"] <= c < sp["max_bet"])
                     or (a == "allin" and (("min_bet" in sp and c == sp["max_bet"]) or
                                           ("min_bet" not in sp and "call" in sp and c == sp["call"]))))
            if not legal:
                illegal += 1
                if illegal <= 3:
                    print("  illegal", a, c, sp)
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
    T.check("served actions are legal in FastHand", illegal == 0, "rows=%d illegal=%d; %s" % (rows, illegal, opp.rates()))


if __name__ == "__main__":
    main()
    T.finish()
