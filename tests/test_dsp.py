"""The scope's spectrum maths (``static/dsp.js``), checked under Node when it is installed."""

import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_dsp_js():
    script = Path(__file__).with_name("dsp_check.js")
    r = subprocess.run([NODE, str(script)], capture_output=True, text=True, timeout=120)
    print(r.stdout)
    assert r.returncode == 0, r.stdout + r.stderr
