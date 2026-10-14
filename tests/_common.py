"""Shared audit plumbing.

Every audit prints PASS/FAIL lines and exits non-zero on any FAIL. Each check
that can pass vacuously has a NEGATIVE CONTROL: the same check run against a
deliberately broken variant, which must FAIL for the audit to pass.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# DONKSOLVER_TEST_INSTALLED=1 tests the INSTALLED package instead of the
# checkout: the source tree has no native library unless one was built into it.
if os.environ.get("DONKSOLVER_TEST_INSTALLED") != "1":
    sys.path.insert(0, os.path.join(ROOT, "src"))
ARTIFACTS = os.environ.get("DONKSOLVER_TEST_ARTIFACTS", os.path.join(ROOT, "artifacts"))
THREADS = int(os.environ.get("DONKSOLVER_TEST_THREADS", "2"))

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok)))
    print("%s  %s  %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    return ok


def control(name, fired, detail=""):
    """A negative control passes when the broken variant was DETECTED."""
    return check("control: " + name, fired, detail)


def finish():
    bad = [n for n, ok in _results if not ok]
    print("\n%d/%d checks passed" % (len(_results) - len(bad), len(_results)))
    if bad:
        print("failed:", *bad, sep="\n  ")
        sys.exit(1)


def reference_engine():
    """pokerfast: the independent engine, rake and evaluator the audits compare against."""
    import pokerfast
    from pokerfast import rake
    return pokerfast, rake


def abstraction_path(buckets="50_50_50"):
    """The audits' card abstraction, built on first use (about a minute)."""
    p = os.path.join(ARTIFACTS, "abs_%s.dab" % buckets)
    if not os.path.exists(p):
        from donksolver import Abstraction
        os.makedirs(ARTIFACTS, exist_ok=True)
        print("  [fixture] building %s" % p, flush=True)
        ab = Abstraction.build([int(b) for b in buckets.split("_")], threads=THREADS)
        tmp = "%s.%d.tmp" % (p, os.getpid())
        ab.save(tmp)
        os.replace(tmp, p)
    return p
