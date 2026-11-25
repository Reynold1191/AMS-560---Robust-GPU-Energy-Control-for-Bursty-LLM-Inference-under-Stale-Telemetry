"""Run every local test, then build the stale-delay sweep figure.

Usage: python run_tests.py
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    code = pytest.main(["-q", "tests"])
    if code != 0:
        print("Tests failed, figure not generated.")
        return int(code)

    from experiments import stale_delay_sweep
    stale_delay_sweep.main([])
    return 0


if __name__ == "__main__":
    sys.exit(main())
