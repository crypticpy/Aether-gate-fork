#
# Aether-gate — a wedged driver must not be able to hold the exit.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""SIGTERM must stop the gate even when adapter.close() never returns.

Measured 2026-08-31 on an RSPdx that had left the USB bus: SIGTERM was
delivered and "bye" was logged, then the process sat in adapter.close() for over
three minutes, because SoapySDRPlay3's stream teardown does not return for a
device that is no longer there. Two further SIGTERMs did nothing — the main
thread was blocked inside a C call, and a Python signal handler only runs
between bytecodes. It took SIGKILL, which skips ReleaseDevice and leaves the
SDRplay API service holding a stale device.

This runs the real entry point in a subprocess, because the fix ends in
os._exit() and there is no honest way to assert that in-process.

Run:  python -m pytest aether_gate/tests/test_shutdown_watchdog.py
"""
import os
import signal
import socket
import subprocess
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A gate whose adapter.close() blocks forever, with the grace cut down so the
# test is quick. Everything else is the shipping code path.
SCRIPT = """
import sys, time
import aether_gate.__main__ as M
from aether_gate.adapters.sim import SimAdapter
M.SHUTDOWN_GRACE_S = {grace}
class Hanging(SimAdapter):
    def close(self):
        while True:
            time.sleep(3600)
M.build_adapter = lambda name, args: Hanging()
sys.exit(M.main(["--adapter", "sim", "--port", "{port}", "--ctl-port", "0"]))
"""


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="no SIGTERM here")
def test_sigterm_wins_over_a_driver_that_never_returns(tmp_path):
    grace = 2.0
    script = tmp_path / "hang.py"
    script.write_text(SCRIPT.format(grace=grace, port=_free_port()))
    env = dict(os.environ, PYTHONPATH=REPO, PYTHONUNBUFFERED="1")
    p = subprocess.Popen([sys.executable, str(script)], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:                 # wait for it to be up
            if p.poll() is not None:
                pytest.fail("gate exited before it served: " + p.communicate()[0])
            time.sleep(0.2)
            if time.monotonic() > deadline - 17.0:
                break
        p.send_signal(signal.SIGTERM)                      # exactly ONE, as a supervisor sends
        # Generous ceiling: the point is that it terminates at all, unassisted.
        out = p.communicate(timeout=grace + 15.0)[0]
    except subprocess.TimeoutExpired:
        p.kill()
        pytest.fail("adapter.close() held the exit — the watchdog did not fire")
    assert p.returncode == 0, (
        f"forced stop returned {p.returncode}; it must be 0 so a supervisor "
        f"running Restart=on-failure does not bounce straight back into the "
        f"same wedged driver")
    assert "cleanup did not finish" in out, out
