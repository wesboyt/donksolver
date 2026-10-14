"""Run every audit; exit non-zero if any fails.

    python tests/run_all.py            # all
    python tests/run_all.py --fast     # skip the slow solver-convergence audit

The card abstraction and the small solved policies the audits read are built
on first use and cached under artifacts/. The differential audits need
pokerfast (`pip install donksolver[test]`). `pytest` runs the same audits.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AUDITS = [
    "audit_eval.py",
    "audit_payoff.py",
    "audit_deal_law.py",
    "audit_abstraction.py",
    "audit_cs_regret.py",
    "audit_average.py",
    "audit_convergence.py",
    "audit_translation.py",
    "audit_serving.py",
]
SLOW = ("audit_convergence.py",)


def run(audit):
    """Returns (exit code, stdout, stderr) of one audit run as its own process."""
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, audit)], cwd=HERE,
                       capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def main():
    fast = "--fast" in sys.argv
    failed = []
    for a in AUDITS:
        if fast and a in SLOW:
            continue
        t0 = time.time()
        code, out, err = run(a)
        tail = [l for l in out.splitlines() if l.startswith(("PASS", "FAIL")) or "checks passed" in l]
        print("=== %s (%.0fs) exit=%d" % (a, time.time() - t0, code))
        for l in tail:
            print("   " + l)
        if code:
            failed.append(a)
            print(err[-2000:])
    print("\n%s" % ("ALL AUDITS PASSED" if not failed else "FAILED: " + ", ".join(failed)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
