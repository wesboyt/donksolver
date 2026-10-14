"""Multiway postflop is part of the tree, is solved, and is served.

Three players, 15bb, one size per street, on the nested abstraction. The same
game is built twice: with postflop_max_players=3 (every flop is played out) and
with postflop_max_players=2 (a three-way flop ends in a check-down leaf).

  1. the full tree has no check-down terminals and has three-way flop, turn
     and river decision nodes; the cut tree has check-down terminals and none
  2. the three-way nodes are solved, not just present: facing a bet three-way
     on the flop, the weakest equity bin folds more than the strongest one
  3. whole three-handed hands on pokerfast's FastHand with every seat served:
     every action is legal and no row comes back no_node, including rows
     where three players are still in on the flop, turn and river
     control: the cut tree's policy returns no_node on the same kind of hands
"""
import os
import random

import numpy as np

import _common as T
import audit_nested
from donksolver import Game, Solve, Solver, write_manifest
from donksolver.engine import Player

STACK_BB = 15


def game(ab, postflop_max_players):
    return Game(players=3, stack_bb=STACK_BB, buckets=ab,
                menus=[[[1.0]], [[0.75]], [[0.75]], [[0.75]]], max_raises=[2, 1, 1, 1],
                mw_max_raises=[2, 1, 1, 1], postflop_max_players=postflop_max_players)


def solved(name, g, abs_path, iterations):
    d = os.path.join(T.ARTIFACTS, "test", name)
    s = Solver.nlhe(g, abs_path, Solve(algo="es", avg_mode="sampled", p_avg=0.1, seed=11))
    st = os.path.join(d, "state.bin")
    if os.path.exists(st):
        os.remove(st)
    pol = s.solve(d, iterations, threads=T.THREADS, chunk=iterations, log=lambda m: None)
    man = os.path.join(d, "bundle.json")
    write_manifest(man, abs_path, [pol], note="test fixture")
    os.remove(st)
    return s, man


def walk(s):
    """Every decision node: (id, node dict)."""
    out, stack, seen = [], [0], set()
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        nd = s.node(nid)
        if nd["actor"] < 0:
            continue
        out.append((nid, nd))
        stack.extend(nd["children"])
    return out


def play(man, hands, rng):
    from pokerfast import FastHand
    opp = Player(man, log=None)
    chips = STACK_BB * 200
    illegal = no_node = rows = 0
    three_way = [0, 0, 0, 0]
    for _ in range(hands):
        h = FastHand(player_count=3, min_stack=chips, max_stack=chips, rng=random.Random(rng.randrange(1 << 30)))
        while not h.done:
            sp = h.get_action_space()
            acts, sizes, _, status = opp.act([h])
            a, c = acts[0], sizes[0]
            rows += 1
            street = h.state.street_index
            if sum(not f for f in h._folded) == 3:
                three_way[street] += 1
            if status[0] >= 2:
                no_node += 1
                h.call() if "call" in sp else h.check()
                continue
            legal = ((a == "fold" and "call" in sp) or (a == "check" and "check" in sp)
                     or (a == "call" and "call" in sp and c == sp["call"])
                     or (a == "raise" and "min_bet" in sp and sp["min_bet"] <= c < sp["max_bet"])
                     or (a == "allin" and (("min_bet" in sp and c == sp["max_bet"]) or
                                           ("min_bet" not in sp and "call" in sp and c == sp["call"]))))
            illegal += not legal
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
    return rows, illegal, no_node, three_way


def main():
    abs_path = audit_nested.path()
    from donksolver import Abstraction
    ab = Abstraction.load(abs_path).buckets
    full, man_full = solved("mw3_full", game(ab, 3), abs_path, 400000)
    cut, man_cut = solved("mw3_cut", game(ab, 2), abs_path, 100000)

    fi, ci = full.info(), cut.info()
    nodes_full, nodes_cut = walk(full), walk(cut)
    mw = lambda nodes: [sum(1 for _, nd in nodes if nd["street"] == s and nd["in_hand"] == 3) for s in (1, 2, 3)]
    T.check("full tree: no check-down terminals, three-way nodes on every street",
            fi["checkdown_terminals"] == 0 and min(mw(nodes_full)) > 0,
            "three-way decision nodes flop/turn/river=%s" % mw(nodes_full))
    T.control("cut tree ends three-way flops in a check-down", ci["checkdown_terminals"] > 0 and max(mw(nodes_cut)) == 0,
              "check-down terminals=%d three-way nodes=%s" % (ci["checkdown_terminals"], mw(nodes_cut)))

    span = audit_nested.BINS[0] * audit_nested.SUB
    top = (audit_nested.BINS[0] - 1) * audit_nested.SUB
    weak, strong = [], []
    for nid, nd in nodes_full:
        if nd["street"] != 1 or nd["in_hand"] != 3 or 0 not in nd["kinds"]:
            continue
        f = nd["kinds"].index(0)
        for cls in range(ab[1] // span):
            weak.append(np.mean([full.strategy(nid, cls * span + k)[f] for k in range(audit_nested.SUB)]))
            strong.append(np.mean([full.strategy(nid, cls * span + top + k)[f] for k in range(audit_nested.SUB)]))
    T.check("three-way flop nodes are solved: the weakest equity bin folds more than the strongest",
            len(weak) > 0 and np.mean(weak) > np.mean(strong) + 0.2,
            "nodes x board classes=%d fold weakest=%.3f strongest=%.3f" % (len(weak), np.mean(weak), np.mean(strong)))

    rng = random.Random(41)
    rows, illegal, no_node, tw = play(man_full, 1500, rng)
    T.check("three-handed hands: every action legal, no no_node rows", illegal == 0 and no_node == 0 and min(tw[1:]) > 0,
            "rows=%d illegal=%d no_node=%d three-way rows flop/turn/river=%s" % (rows, illegal, no_node, tw[1:]))
    rows_c, _, no_node_c, _ = play(man_cut, 1500, rng)
    T.control("cut tree's policy returns no_node on three-way flops", no_node_c > 0,
              "rows=%d no_node=%d" % (rows_c, no_node_c))


if __name__ == "__main__":
    main()
    T.finish()
