"""Make the vendored ``reBotArm_control_py`` importable.

Upstream has no packaging metadata that survives ``pip install`` (its flat
layout trips setuptools' auto-discovery, and it resolves ``config/`` and
``urdf/`` relative to its own source tree), so it is used the way its own
examples use it: the checkout directory is put on ``sys.path``.

Resolution order:

1. an already importable ``reBotArm_control_py`` (e.g. installed by hand);
2. ``$REBOTARM_CONTROL_PY_DIR`` if set;
3. the git submodule at ``third_party/reBotArm_control_py``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUBMODULE_DIR = PROJECT_ROOT / "third_party" / "reBotArm_control_py"
ENV_VAR = "REBOTARM_CONTROL_PY_DIR"


def upstream_dir() -> Path:
    return Path(os.environ[ENV_VAR]).expanduser() if os.environ.get(ENV_VAR) else SUBMODULE_DIR


def ensure_importable() -> Path | None:
    """Put the upstream checkout on ``sys.path``; return the directory used (None if pre-installed)."""
    if importlib.util.find_spec("reBotArm_control_py") is not None:
        return None
    d = upstream_dir()
    if not (d / "reBotArm_control_py" / "__init__.py").exists():
        raise ImportError(
            f"reBotArm_control_py not found at {d}. Initialise the submodule with "
            "'git submodule update --init' or point REBOTARM_CONTROL_PY_DIR at a checkout."
        )
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    return d
