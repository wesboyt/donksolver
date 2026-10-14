"""Frozen policy bundles and the batch action server.

A bundle is a manifest (JSON) naming one abstraction file and one policy file
per stack class. It is immutable once written: `version` is a hash over every
member's content, so it identifies the strategy being served.
"""
import ctypes as C
import hashlib
import json
import os

import numpy as np

from . import _native as N

KINDS = ("fold", "check", "call", "raise", "allin")
STATUS = ("ok", "offtree", "no_node", "desync", "bad_input")
ST_OK, ST_OFFTREE, ST_NO_NODE, ST_DESYNC, ST_BAD_INPUT = range(5)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_manifest(path, abstraction, policies, commit_frac=0.9, note=""):
    """Create a bundle manifest. Paths are stored relative to the manifest."""
    base = os.path.dirname(os.path.abspath(path))
    rel = lambda p: os.path.relpath(os.path.abspath(p), base).replace(os.sep, "/")
    members = [{"path": rel(abstraction), "sha256": _sha(abstraction)}]
    pols = [{"path": rel(p), "sha256": _sha(p)} for p in policies]
    digest = hashlib.sha256(json.dumps([members, pols, commit_frac], sort_keys=True).encode()).hexdigest()
    man = {"format": "donksolver-bundle/1", "abstraction": members[0], "policies": pols,
           "commit_frac": commit_frac, "version": digest[:16], "note": note}
    with open(path, "w") as fh:
        json.dump(man, fh, indent=2)
    return man


class Bundle:
    """Serves actions for real game states. Thread-safe (read-only native state)."""

    def __init__(self, manifest_path, verify=True):
        with open(manifest_path) as fh:
            man = json.load(fh)
        if man.get("format") != "donksolver-bundle/1":
            raise ValueError("not a donksolver bundle manifest: %s" % manifest_path)
        base = os.path.dirname(os.path.abspath(manifest_path))
        full = lambda m: os.path.join(base, m["path"])
        if verify:
            for m in [man["abstraction"]] + man["policies"]:
                if _sha(full(m)) != m["sha256"]:
                    raise ValueError("bundle member changed since the manifest was written: %s" % m["path"])
        self.manifest = man
        self.version = man["version"]
        self.commit_frac = float(man.get("commit_frac", 0.9))
        paths = "\n".join(full(m) for m in man["policies"])
        h = N.lib.ds_bundle_load(os.fsencode(full(man["abstraction"])), paths.encode(), self.commit_frac)
        N.check(h)
        self._h = h
        self.classes = []
        buf = (C.c_int64 * 8)()
        n = N.lib.ds_bundle_info(h, 0, buf)
        for k in range(n):
            N.lib.ds_bundle_info(h, k, buf)
            self.classes.append({"players": buf[0], "stack": buf[1], "sb": buf[2], "bb": buf[3],
                                 "iterations": buf[4], "version": buf[5] & 0xFFFFFFFFFFFFFFFF,
                                 "nodes": buf[6], "entries": buf[7]})
        self.players = self.classes[0]["players"]

    def __del__(self):
        h, self._h = getattr(self, "_h", None), None
        if h:
            N.lib.ds_bundle_free(h)

    def act(self, stacks, blinds, hole, board, histories, u):
        """Batch query.

        stacks    (R, n) int starting stacks
        blinds    (R, 2) int (sb, bb)
        hole      (R, 2) acting seat's hole card ids
        board     list of R lists of board card ids
        histories list of R lists of (seat, is_fold, chips) -- chips ADDITIONAL, 0 = check
        u         (R,) uniforms in [0,1) for the action draw
        Returns a dict of arrays: kind, chips, prob, status, cls, node, bucket, offtree.
        """
        R = len(histories)
        n = self.players
        meta = np.zeros((R, 4), dtype=np.int32)
        st = np.zeros((R, 6), dtype=np.int64)
        st[:, :n] = np.asarray(stacks, dtype=np.int64).reshape(R, n)
        bl = np.ascontiguousarray(np.asarray(blinds, dtype=np.int64).reshape(R, 2))
        cards = np.zeros((R, 7), dtype=np.int32)
        cards[:, :2] = np.asarray(hole, dtype=np.int32).reshape(R, 2)
        total = sum(len(h) for h in histories)
        hs = np.zeros(max(total, 1), dtype=np.int32)
        hf = np.zeros(max(total, 1), dtype=np.int32)
        hc = np.zeros(max(total, 1), dtype=np.int64)
        off = 0
        for r in range(R):
            b = board[r]
            cards[r, 2:2 + len(b)] = b
            h = histories[r]
            meta[r] = (n, len(b), off, len(h))
            for k, (s, f, c) in enumerate(h):
                hs[off + k] = s
                hf[off + k] = f
                hc[off + k] = c
            off += len(h)
        uu = np.ascontiguousarray(np.asarray(u, dtype=np.float64).reshape(R))
        out = {k: np.zeros(R, dtype=t) for k, t in (("kind", np.int32), ("chips", np.int64), ("prob", np.float64),
                                                    ("status", np.int32), ("cls", np.int32), ("node", np.int64),
                                                    ("bucket", np.int32), ("offtree", np.int32))}
        N.check(N.lib.ds_bundle_act(
            self._h, R, N.ptr(meta, C.c_int32), N.ptr(st, C.c_int64), N.ptr(bl, C.c_int64),
            N.ptr(cards, C.c_int32), N.ptr(hs, C.c_int32), N.ptr(hf, C.c_int32), N.ptr(hc, C.c_int64),
            N.ptr(uu, C.c_double), N.ptr(out["kind"], C.c_int32), N.ptr(out["chips"], C.c_int64),
            N.ptr(out["prob"], C.c_double), N.ptr(out["status"], C.c_int32), N.ptr(out["cls"], C.c_int32),
            N.ptr(out["node"], C.c_int64), N.ptr(out["bucket"], C.c_int32), N.ptr(out["offtree"], C.c_int32)))
        return out

    def strategy(self, cls, node, bucket):
        p = (C.c_float * 16)()
        k = (C.c_uint8 * 16)()
        f = (C.c_float * 16)()
        nA = N.lib.ds_bundle_strategy(self._h, int(cls), int(node), int(bucket), p, k, f)
        N.check(nA >= 0)
        return list(p)[:nA], list(k)[:nA], list(f)[:nA]
