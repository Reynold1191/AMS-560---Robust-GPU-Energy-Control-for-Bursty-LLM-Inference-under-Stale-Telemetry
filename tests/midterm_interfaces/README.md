# Midterm interface tests

These tests were written by each member against the midterm interfaces
(clock ladder up to 3090 MHz, SLO 8.5 s, the first load generator and
scale simulator, the original metrics collector).

The final integrated code changed those interfaces to match the RTX 5060
testbed and the hardware experiment runner, so these tests no longer match it.
They are kept as a record of the midterm work and are not collected by default.
The tests for the final code are in `tests/test_*.py` (run `python run_tests.py`).

To look at them against the midterm code, check out the commit before the
final integration and run `pytest tests/midterm_interfaces`.
