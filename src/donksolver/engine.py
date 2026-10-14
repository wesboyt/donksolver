"""Serve a bundle's actions for hands played on pokerfast's `FastHand` engine.

    from pokerfast import FastHand
    from donksolver.engine import Player

    player = Player("runs/bundle.json")
    hand = FastHand()
    actions, chips, probs, status = player.act([hand])

Only the public state (starting stacks, blinds, board, action history) and the
ACTING seat's own hole cards are read; the solver seat never sees other
players' cards. Actions come back as

    ('fold'|'check'|'call'|'raise'|'allin', chips)

where 'raise' chips are ADDITIONAL, as `FastHand.bet_or_raise` takes them, and
'allin' is a shove when a raise is legal, else an all-in call.

Rows the abstraction cannot answer (multiway postflop, desynced lines) come
back with status >= ST_NO_NODE and no usable action; the caller decides what
plays them, and the rate is counted here.

The bundle is immutable and `version` identifies it. The action draw uses the
caller's uniforms, and the coin that maps an off-tree raise to an abstract size
is a function of the state, so the same state is always read the same way.

pokerfast is not a dependency of donksolver: this module only reads attributes
of the hand objects it is given.
"""
import random
import threading

import numpy as np

from .cards import card_id
from .policy import Bundle, KINDS, STATUS, ST_NO_NODE

_CARD = {}


def _cid(s):
    v = _CARD.get(s)
    if v is None:
        v = _CARD[s] = card_id(s)
    return v


def history_of(hand):
    """[(seat, is_fold, chips)] in action order, from FastHand.u_hand[3..6]."""
    out = []
    for street in hand.u_hand[3:7]:
        for a in street:
            # "p%df" or "p%dc%d"
            j = a.find("f") if a.endswith("f") else a.find("c")
            seat = int(a[1:j])
            if a.endswith("f") and j == len(a) - 1:
                out.append((seat, 1, 0))
            else:
                out.append((seat, 0, int(a[j + 1:])))
    return out


class Player:
    def __init__(self, manifest_path, log_every=200000, log=print):
        self.bundle = Bundle(manifest_path)
        self.version = self.bundle.version
        self._lock = threading.Lock()
        self.counts = np.zeros(len(STATUS), dtype=np.int64)
        self.kinds = np.zeros(len(KINDS), dtype=np.int64)
        self._next_log = log_every
        self._log_every = log_every
        self._log = log

    def describe(self):
        cls = ", ".join("%dbb(%d it)" % (c["stack"] // max(c["bb"], 1), c["iterations"])
                        for c in self.bundle.classes)
        return "donksolver bundle %s: %d players, classes [%s]" % (
            self.version, self.bundle.players, cls)

    def act(self, hands, uniforms=None):
        """Returns (actions, sizes, probs, status) for a batch of FastHands to act."""
        R = len(hands)
        n = self.bundle.players
        stacks = np.empty((R, n), dtype=np.int64)
        blinds = np.empty((R, 2), dtype=np.int64)
        hole = np.empty((R, 2), dtype=np.int32)
        boards, hists = [], []
        for r, h in enumerate(hands):
            st = h.state
            if len(st.starting_stacks) != n:
                raise ValueError("bundle is for %d players, hand has %d" % (n, len(st.starting_stacks)))
            stacks[r] = st.starting_stacks
            blinds[r] = (h.small_blind, h.big_blind)
            hc = h._hole[st.turn_index]
            hole[r] = (_cid(hc[0:2]), _cid(hc[2:4]))
            boards.append([_cid(c) for c in st.board])
            hists.append(history_of(h))
        u = uniforms if uniforms is not None else [random.random() for _ in range(R)]
        out = self.bundle.act(stacks, blinds, hole, boards, hists, u)
        actions = [KINDS[k] for k in out["kind"]]
        sizes = [int(c) for c in out["chips"]]
        with self._lock:
            self.counts += np.bincount(out["status"], minlength=len(STATUS))
            ok = out["status"] < ST_NO_NODE
            self.kinds += np.bincount(out["kind"][ok], minlength=len(KINDS))
            tot = int(self.counts.sum())
            if self._log and tot >= self._next_log:
                self._next_log += self._log_every
                self._log("  [donksolver %s] %s" % (self.version, self.rates()))
        return actions, sizes, out["prob"], out["status"]

    def rates(self):
        tot = max(int(self.counts.sum()), 1)
        s = " ".join("%s=%.1f%%" % (STATUS[i], 100.0 * self.counts[i] / tot) for i in range(len(STATUS)))
        k = max(int(self.kinds.sum()), 1)
        s += " | " + " ".join("%s=%.1f%%" % (KINDS[i], 100.0 * self.kinds[i] / k) for i in range(len(KINDS)))
        return "rows=%d %s" % (tot, s)
