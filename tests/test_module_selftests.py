"""Run the built-in self-tests of the modules that do not need a GPU."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("module", [
    "slo_budget",
    "delay_buffer",
    "controllers.naive_slo",
    "controllers.robust_delay_budget",
])
def test_module_selftest(module):
    # The self-tests print non-ASCII characters, which a cp1252 Windows console cannot encode.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore::RuntimeWarning"}
    result = subprocess.run([sys.executable, "-m", module], cwd=os.path.join(ROOT, "src"), env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
