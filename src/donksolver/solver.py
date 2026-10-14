"""Abstraction and Solver: thin, typed wrappers over the native core."""
import ctypes as C
import json
import math
import os
import time

import numpy as np

from . import _native as N
from .config import Game, Solve


BOARD_MODES = ("none", "suits", "texture")


class Abstraction:
    """Card abstraction file (.dab): 169 preflop classes plus postflop buckets.

    kind="percentile": `buckets` per street, a global percentile ranking of
    E[HS^2] (flop, turn) and HS (river).

    kind="nested": on every flop and turn board, `buckets` equal-width equity
    bins between that board's weakest and strongest holding, each split into
    `sub` bins by potential; `buckets[2]` equal-width HS bins on the river.
    `boards` adds a board class per street to the key ("none", "suits" = flush
    structure, "texture" = flush structure x high-card counts).

    `buckets` (the property) is the number of buckets per street as the solver
    sees them, board classes and sub-bins included.
    """

    def __init__(self, handle, path=None):
        self._h = handle
        self.path = path

    @classmethod
    def build(cls, buckets=(50, 50, 50), threads=2, river_boards=20000, seed=1, progress=None,
              kind="percentile", sub=4, boards=("none", "none", "none")):
        def _p(stage, frac):
            if progress:
                progress(stage.decode(), frac)
        cb = N.PROGRESS(_p)
        if kind == "percentile":
            h = N.lib.ds_abs_build(int(buckets[0]), int(buckets[1]), int(buckets[2]), int(threads),
                                   int(river_boards), int(seed), cb)
        elif kind == "nested":
            if isinstance(boards, str):
                boards = (boards,) * 3
            bm = [BOARD_MODES.index(b) for b in boards]
            h = N.lib.ds_abs_build_nested(int(buckets[0]), int(buckets[1]), int(buckets[2]), int(sub),
                                          bm[0], bm[1], bm[2], int(threads), cb)
        else:
            raise ValueError("kind must be 'percentile' or 'nested'")
        N.check(h)
        return cls(h)

    @property
    def params(self):
        """How the file was built: kind, sub-bins, bins and board mode per street."""
        out = (C.c_int32 * 8)()
        N.lib.ds_abs_params(self._h, out)
        return {"kind": ("percentile", "nested")[out[0]], "sub": out[1], "bins": list(out[2:5]),
                "boards": [BOARD_MODES[m] for m in out[5:8]]}

    @classmethod
    def load(cls, path):
        h = N.lib.ds_abs_load(os.fsencode(path))
        N.check(h)
        return cls(h, path)

    def save(self, path):
        N.check(N.lib.ds_abs_save(self._h, os.fsencode(path)))
        self.path = path

    @property
    def hash(self):
        return int(N.lib.ds_abs_hash(self._h))

    @property
    def buckets(self):
        out = (C.c_int32 * 4)()
        N.lib.ds_abs_buckets(self._h, out)
        return list(out)

    def bucket(self, street, hole, board=()):
        h = (C.c_int32 * 2)(*hole)
        b = (C.c_int32 * 5)(*(list(board) + [0] * (5 - len(board))))
        r = N.lib.ds_abs_bucket(self._h, int(street), h, b)
        if r < 0:
            raise N.NativeError(N.lib.ds_last_error().decode())
        return r

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            N.lib.ds_abs_free(h)


def tree_info(config_text):
    buf = C.create_string_buffer(1 << 14)
    N.check(N.lib.ds_tree_info(config_text.encode(), buf, len(buf)) >= 0)
    return _parse_info(buf.value.decode())


def _parse_info(text):
    out = {}
    for line in text.strip().splitlines():
        k, v = line.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            out[k] = float(v)
    return out


class Solver:
    """One abstract game + one solve. The README describes the algorithms."""

    def __init__(self, config_text):
        self.config_text = config_text
        h = N.lib.ds_solver_create(config_text.encode())
        N.check(h)
        self._h = h
        self.players = self.info()["players"]

    @classmethod
    def nlhe(cls, game: Game, abstraction_path, solve: Solve = None):
        solve = solve or Solve()
        text = game.text() + solve.text() + "abstraction=%s\n" % os.path.abspath(abstraction_path)
        return cls(text)

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            N.lib.ds_solver_free(h)

    def info(self):
        buf = C.create_string_buffer(1 << 14)
        N.check(N.lib.ds_solver_info(self._h, buf, len(buf)) >= 0)
        return _parse_info(buf.value.decode())

    @property
    def iterations(self):
        return int(N.lib.ds_solver_iterations(self._h))

    @property
    def entries(self):
        return int(N.lib.ds_solver_entries(self._h))

    def run(self, iterations, threads=1):
        N.check(N.lib.ds_solver_run(self._h, int(iterations), int(threads)))

    def root_ev(self):
        """Per seat: (mean, standard error, n) of the one-seat-per-deal root value."""
        s = (C.c_double * 6)()
        q = (C.c_double * 6)()
        n = (C.c_uint64 * 6)()
        k = N.lib.ds_solver_root_ev(self._h, s, q, n)
        out = []
        for i in range(k):
            if n[i] < 2:
                out.append((float("nan"), float("nan"), int(n[i])))
                continue
            m = s[i] / n[i]
            var = max(q[i] / n[i] - m * m, 0.0) * n[i] / (n[i] - 1)
            out.append((m, math.sqrt(var / n[i]), int(n[i])))
        return out

    def reset_ev(self):
        N.lib.ds_solver_reset_ev(self._h)

    def reset_average(self):
        N.lib.ds_solver_reset_average(self._h)

    def exact(self):
        """Enumerable games: (ev, br) per seat of the AVERAGE profile. gain = br - ev."""
        ev = (C.c_double * 6)()
        br = (C.c_double * 6)()
        k = N.lib.ds_solver_exact(self._h, ev, br)
        N.check(k)
        return list(ev)[:k], list(br)[:k]

    def exploitability(self):
        ev, br = self.exact()
        gains = [b - e for e, b in zip(ev, br)]
        return {"ev": ev, "br": br, "gains": gains, "nashconv": sum(gains), "max_gain": max(gains)}

    def node(self, node_id):
        info = (C.c_int32 * 6)()
        ch = (C.c_uint32 * 16)()
        kinds = (C.c_uint8 * 16)()
        fr = (C.c_float * 16)()
        chips = (C.c_int64 * 16)()
        nA = N.lib.ds_solver_node(self._h, int(node_id), info, ch, kinds, fr, chips)
        N.check(nA >= 0)
        return {"actor": info[0], "street": info[1], "nA": info[2], "term": info[3], "in_hand": info[5],
                "children": list(ch)[:nA], "kinds": list(kinds)[:nA], "fracs": list(fr)[:nA],
                "chips": list(chips)[:nA]}

    def strategy(self, node_id, bucket, average=True):
        out = (C.c_double * 16)()
        nA = N.lib.ds_solver_strategy(self._h, int(node_id), int(bucket), int(bool(average)), out)
        N.check(nA >= 0)
        return list(out)[:nA]

    def debug_regret(self, player, exact=True, samples=0, broken=False, seed=1):
        out = np.zeros(self.entries, dtype=np.float64)
        N.check(N.lib.ds_solver_debug_regret(self._h, int(player), 0 if exact else 1, int(samples),
                                             int(broken), int(seed), N.ptr(out, C.c_double)))
        return out

    def shadow(self, on=True, broken=False):
        N.check(N.lib.ds_solver_shadow(self._h, int(on), int(broken)))

    def shadow_compare(self, min_mass=0.0):
        wm = C.c_double()
        worst = N.lib.ds_solver_shadow_compare(self._h, float(min_mass), C.byref(wm))
        return worst, wm.value

    def save_state(self, path):
        tmp = path + ".tmp"
        N.check(N.lib.ds_solver_save_state(self._h, os.fsencode(tmp)))
        os.replace(tmp, path)

    def load_state(self, path):
        N.check(N.lib.ds_solver_load_state(self._h, os.fsencode(path)))

    def save_policy(self, path):
        tmp = path + ".tmp"
        N.check(N.lib.ds_solver_save_policy(self._h, os.fsencode(tmp)))
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    def solve(self, out_dir, iterations, threads=1, chunk=None, checkpoint_every_s=900,
              time_limit_s=None, log=print, reset_ev_each_chunk=True):
        """Checkpointed solve loop.

        Writes out_dir/state.bin (resumable), out_dir/log.jsonl (per chunk: iterations,
        rate, Iterations/Nodes, root EV per seat with SE) and, at the end, policy.dpol.
        Resumes from state.bin when present. The per-chunk root EV is estimated on that
        chunk's deals only (reset_ev_each_chunk) so drift is visible; that drift is
        the doneness read. Iterations/Nodes is logged for comparison only: it is a
        budget, not a regret guarantee.
        """
        os.makedirs(out_dir, exist_ok=True)
        state = os.path.join(out_dir, "state.bin")
        with open(os.path.join(out_dir, "config.txt"), "w") as fh:
            fh.write(self.config_text)
        if os.path.exists(state):
            self.load_state(state)
            log("resumed at iteration %d" % self.iterations)
        info = self.info()
        rows = sum(info.get("rows.%d" % s, 0) for s in range(4))
        chunk = chunk or max(10000, min(2000000, iterations // 50 or 1))
        t_start = time.time()
        last_ckpt = time.time()
        logf = open(os.path.join(out_dir, "log.jsonl"), "a")
        try:
            while self.iterations < iterations:
                if time_limit_s and time.time() - t_start > time_limit_s:
                    log("time limit reached")
                    break
                n = min(chunk, iterations - self.iterations)
                if reset_ev_each_chunk:
                    self.reset_ev()
                t0 = time.time()
                self.run(n, threads)
                dt = time.time() - t0
                ev = self.root_ev()
                rec = {"t": time.time(), "iterations": self.iterations, "it_per_s": n / max(dt, 1e-9),
                       "iterations_per_node": self.iterations / max(rows, 1),
                       "root_ev": [e[0] for e in ev], "root_ev_se": [e[1] for e in ev]}
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                log("it=%d  %.0f it/s  I/N=%.2f  root EV(chips) %s" % (
                    self.iterations, rec["it_per_s"], rec["iterations_per_node"],
                    " ".join("%.1f+-%.1f" % (m, s) for m, s, _ in ev)))
                if time.time() - last_ckpt > checkpoint_every_s:
                    self.save_state(state)
                    last_ckpt = time.time()
        finally:
            logf.close()
            self.save_state(state)
        pol = os.path.join(out_dir, "policy.dpol")
        self.save_policy(pol)
        return pol
