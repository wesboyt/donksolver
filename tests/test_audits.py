"""pytest entry point: each audit runs as its own process and must exit 0.

The audits are scripts (tests/audit_*.py) that print one PASS/FAIL line per
check; a failure shows that output.
"""
import pytest

import _common as T
import run_all


@pytest.fixture(scope="session", autouse=True)
def abstraction():
    # Built once here, so the audits that share it do not each wait for it.
    return T.abstraction_path()


@pytest.mark.parametrize("audit", [
    pytest.param(a, marks=pytest.mark.slow) if a in run_all.SLOW else a for a in run_all.AUDITS])
def test_audit(audit):
    code, out, err = run_all.run(audit)
    assert code == 0, "%s failed\n%s\n%s" % (audit, out[-4000:], err[-4000:])
