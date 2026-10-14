"""donksolver command line.

  donksolver abstraction --buckets 50,50,50 --threads 2 --out abs.dab
  donksolver abstraction --kind nested --buckets 12,12,24 --sub 4 --out nested.dab
  donksolver size --preset 6max-lite --stack-bb 100
  donksolver solve --preset 6max-lite --stack-bb 100 --abstraction abs.dab \
        --iterations 50000000 --threads 2 --out runs/6max_100
  donksolver bundle --abstraction abs.dab --out bundle.json runs/*/policy.dpol
  donksolver toy --game leduc --players 2 --iterations 1000000
"""
import argparse
import glob
import json
import sys
import time

from . import config as cfg
from .solver import Abstraction, Solver, tree_info
from .policy import write_manifest


def _ints(s):
    return [int(x) for x in s.split(",")]


def cmd_abstraction(a):
    t0 = time.time()
    last = [0.0]

    def prog(stage, frac):
        if time.time() - last[0] > 10 or frac >= 1.0:
            last[0] = time.time()
            print("%7.1fs %s %.3f" % (time.time() - t0, stage, frac), flush=True)

    b = _ints(a.buckets)
    ab = Abstraction.build(b, threads=a.threads, river_boards=a.river_boards, seed=a.seed, progress=prog,
                           kind=a.kind, sub=a.sub, boards=tuple(a.boards.split(",")) if "," in a.boards else a.boards)
    ab.save(a.out)
    print("saved %s buckets=%s hash=%016x (%.1fs)" % (a.out, ab.buckets, ab.hash, time.time() - t0))


def _game(a):
    if a.buckets:
        buckets = _ints(a.buckets)
    elif getattr(a, "abstraction", None):
        buckets = Abstraction.load(a.abstraction).buckets[1:]
    else:
        buckets = [50, 50, 50]
    g = cfg.preset(a.preset, a.stack_bb, buckets)
    if a.postflop_max_players is not None:
        g.postflop_max_players = a.postflop_max_players
    if a.rake_cap is not None:
        g.rake_cap = a.rake_cap
    return g


def cmd_size(a):
    info = tree_info(_game(a).text())
    for k, v in info.items():
        print("%-22s %s" % (k, v))
    print("tables (double) %.2f GB, (float) %.2f GB" % (info["entries"] * 16 / 2**30, info["entries"] * 8 / 2**30))


def cmd_solve(a):
    solve = cfg.Solve(algo=a.algo, weighting=a.weighting, avg_mode=a.avg_mode, p_avg=a.p_avg,
                      precision=a.precision, seed=a.seed, memory_limit_gb=a.memory_limit_gb)
    s = Solver.nlhe(_game(a), a.abstraction, solve)
    info = s.info()
    print("tree: nodes=%d entries=%d tables=%.2f GB" % (info["nodes"], info["entries"], info["table_bytes"] / 2**30))
    pol = s.solve(a.out, a.iterations, threads=a.threads, chunk=a.chunk, checkpoint_every_s=a.checkpoint_s,
                  time_limit_s=a.time_limit_s)
    print("policy", pol)


def cmd_bundle(a):
    pols = []
    for p in a.policies:
        pols.extend(sorted(glob.glob(p)) or [p])
    man = write_manifest(a.out, a.abstraction, pols, commit_frac=a.commit_frac, note=a.note)
    print(json.dumps(man, indent=2))


def cmd_toy(a):
    s = Solver(cfg.toy(a.game, a.players, algo=a.algo, weighting=a.weighting, avg_mode="weighted",
                       p_avg=1.0, seed=a.seed))
    done = 0
    step = max(1, a.iterations // 10)
    while done < a.iterations:
        s.run(step, a.threads)
        done += step
        x = s.exploitability()
        print("it=%-10d nashconv=%.6f gains=%s" % (done, x["nashconv"], ["%.5f" % g for g in x["gains"]]))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="donksolver")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("abstraction")
    p.add_argument("--buckets", default="50,50,50")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--river-boards", type=int, default=20000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--kind", default="percentile", choices=["percentile", "nested"])
    p.add_argument("--sub", type=int, default=4, help="nested: potential sub-bins per equity bin")
    p.add_argument("--boards", default="none",
                   help="nested: board class, one value or flop,turn,river (none | suits | texture)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_abstraction)

    for name, fn in (("size", cmd_size), ("solve", cmd_solve)):
        p = sub.add_parser(name)
        p.add_argument("--preset", default="6max-lite")
        p.add_argument("--stack-bb", type=float, default=100.0)
        p.add_argument("--buckets", default="")
        p.add_argument("--rake-cap", type=int, default=None)
        p.add_argument("--postflop-max-players", type=int, default=None,
                       help="players that may see a flop before the tree ends in a check-down")
        if name == "size":
            p.add_argument("--abstraction", default=None, help="read the bucket counts from this file")
        if name == "solve":
            p.add_argument("--abstraction", required=True)
            p.add_argument("--iterations", type=int, required=True)
            p.add_argument("--threads", type=int, default=2)
            p.add_argument("--chunk", type=int, default=None)
            p.add_argument("--checkpoint-s", type=float, default=900)
            p.add_argument("--time-limit-s", type=float, default=None)
            p.add_argument("--algo", default="es")
            p.add_argument("--weighting", default="linear")
            p.add_argument("--avg-mode", default="sampled")
            p.add_argument("--p-avg", type=float, default=0.02)
            p.add_argument("--precision", default="double")
            p.add_argument("--memory-limit-gb", type=float, default=12.0)
            p.add_argument("--seed", type=int, default=1)
            p.add_argument("--out", required=True)
        p.set_defaults(fn=fn)

    p = sub.add_parser("bundle")
    p.add_argument("--abstraction", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--commit-frac", type=float, default=0.9)
    p.add_argument("--note", default="")
    p.add_argument("policies", nargs="+")
    p.set_defaults(fn=cmd_bundle)

    p = sub.add_parser("toy")
    p.add_argument("--game", default="leduc")
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--iterations", type=int, default=1000000)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--algo", default="es")
    p.add_argument("--weighting", default="linear")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(fn=cmd_toy)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
