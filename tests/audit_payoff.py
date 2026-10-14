"""The solver's game is the reference engine's game.

Random hands on pokerfast's FastHand (random 20-500bb stacks, all-in heavy
action so side pots and short all-ins occur). At EVERY step the native replay
of the action history must report the same actor, street, amount to call,
raise bounds and pot; at the end the native settlement (side pots, odd chips,
rake) must reproduce FastHand payoffs raked by pokerfast.rake exactly.

controls: rake without no-flop-no-drop, and exact (non-integer) pot splitting,
must each be detected.
"""
import ctypes as C
import random

import numpy as np

import _common as T
from donksolver import _native as N
from donksolver import rake as drake
from donksolver.cards import card_id
from donksolver.engine import history_of


def replay(hand):
    st = hand.state
    n = len(st.starting_stacks)
    h = history_of(hand)
    stacks = (C.c_int64 * n)(*st.starting_stacks)
    hs = (C.c_int32 * max(len(h), 1))(*[x[0] for x in h])
    hf = (C.c_int32 * max(len(h), 1))(*[x[1] for x in h])
    hc = (C.c_int64 * max(len(h), 1))(*[x[2] for x in h])
    out = (C.c_int64 * 9)()
    com = (C.c_int64 * n)()
    k = N.lib.ds_replay(n, stacks, hand.small_blind, hand.big_blind, hs, hf, hc, len(h), out, com)
    return k, len(h), list(out), list(com)


def main():
    fast_hand, rake = T.reference_engine()
    rng = random.Random(11)
    for bb in (2, 10, 200, 2000):
        for n in (2, 3, 6):
            T.check("rake cap bb=%d n=%d matches pokerfast.rake" % (bb, n), drake.rake_cap(bb, n) == rake.rake_cap(bb, n))
    step_bad = pay_bad = steps = 0
    ctl_nfnd = ctl_split = 0
    hands = 6000
    showdowns = sidepots = 0
    for hidx in range(hands):
        hand = fast_hand.FastHand(rng=random.Random(rng.randrange(1 << 30)))
        while not hand.done:
            sp = hand.get_action_space()
            k, hl, out, com = replay(hand)
            steps += 1
            exp_call = sp.get("call", 0)
            ok = (k == hl and out[0] == 0 and out[3] == sp["player"] and out[2] == hand.state.street_index
                  and out[5] == exp_call and out[4] == hand.pot_size()
                  and bool(out[8]) == ("min_bet" in sp)
                  and (not out[8] or (out[6] == sp["min_bet"] and out[7] == sp["max_bet"])))
            if not ok:
                step_bad += 1
                if step_bad <= 3:
                    print("step mismatch", sp, out, hand.u_hand[3:7])
            r = rng.random()
            if "call" in sp and r < 0.18:
                hand.fold()
            elif "min_bet" in sp and r < 0.45:
                if rng.random() < 0.4:
                    hand.bet_or_raise(sp["max_bet"])
                else:
                    hand.bet_or_raise(rng.randint(sp["min_bet"], sp["max_bet"]))
            elif "call" in sp:
                hand.call()
            else:
                hand.check()
        st = hand.state
        n = len(st.starting_stacks)
        k, hl, out, com = replay(hand)
        folded = (C.c_uint8 * n)(*[1 if f else 0 for f in hand._folded])
        board = [card_id(c) for c in st.board]
        vals = []
        for i in range(n):
            hc = hand._hole[i]
            cards = [card_id(hc[:2]), card_id(hc[2:])] + board
            vals.append(N.lib.ds_eval((C.c_int32 * 7)(*cards), len(cards)) if len(cards) >= 5 else 0)
        value = (C.c_uint32 * n)(*vals)
        committed = (C.c_int64 * n)(*hand._committed)
        alive = n - sum(hand._folded)
        showdown = 1 if alive > 1 else 0
        showdowns += showdown
        sidepots += len(set(c for c, f in zip(hand._committed, hand._folded) if not f)) > 1
        saw_flop = 1 if len(board) >= 3 else 0
        cap = rake.rake_cap(hand.big_blind, n)
        expect = rake.apply_rake_to_payoffs(list(st.payoffs), pot=hand.final_pot, big_blind=hand.big_blind,
                                            n_players_dealt=n, saw_flop=bool(saw_flop))
        o = (C.c_double * n)()
        N.lib.ds_settle(n, committed, folded, value, showdown, saw_flop, rake.RAKE_PCT, cap, 1, 1, o)
        if not (k == hl and out[0] == 1 and list(com) == list(hand._committed)
                and np.allclose(list(o), expect, atol=1e-9)):
            pay_bad += 1
            if pay_bad <= 3:
                print("payoff mismatch", list(o), expect, hand.u_hand)
        o2 = (C.c_double * n)()
        N.lib.ds_settle(n, committed, folded, value, showdown, saw_flop, rake.RAKE_PCT, cap, 0, 1, o2)
        ctl_nfnd += not np.allclose(list(o2), expect, atol=1e-9)
        o3 = (C.c_double * n)()
        N.lib.ds_settle(n, committed, folded, value, showdown, saw_flop, rake.RAKE_PCT, cap, 1, 0, o3)
        ctl_split += not np.allclose(list(o3), expect, atol=1e-9)
    T.check("replay == FastHand at every step", step_bad == 0, "steps=%d mismatches=%d" % (steps, step_bad))
    T.check("settlement+rake == FastHand+pokerfast.rake", pay_bad == 0,
            "hands=%d showdowns=%d side-pot hands=%d mismatches=%d" % (hands, showdowns, sidepots, pay_bad))
    T.control("rake on unflopped pots detected", ctl_nfnd > 0, "hands differing=%d" % ctl_nfnd)
    T.control("exact pot splitting detected", ctl_split > 0, "hands differing=%d" % ctl_split)


if __name__ == "__main__":
    main()
    T.finish()
