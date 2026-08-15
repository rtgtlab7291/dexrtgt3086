"""Regression guard: `src/robokit` must never import the official `smplx`.

`robokit.smplx` (engine + SMPL-X + MANO format adapters) fully replaces
the official `smplx` package; no code in this repo should import it.
"""

import re
import sys
from pathlib import Path
from typing import List

import pytest


class _BlockSmplx:
    """`meta_path` finder that turns `import smplx` into `ImportError`."""

    def find_module(self, name, path=None):
        if name == "smplx" or name.startswith("smplx."):
            return self
        return None

    def load_module(self, name):
        raise ImportError(
            f"Runtime guard: {name!r} import is forbidden. "
            f"The official `smplx` package is no longer used in this repo; use `robokit.smplx` instead."
        )


@pytest.fixture
def block_smplx(monkeypatch: pytest.MonkeyPatch):
    for mod_name in list(sys.modules.keys()):
        if mod_name == "smplx" or mod_name.startswith("smplx."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockSmplx(), *sys.meta_path])


def test_robokit_runtime_imports(block_smplx: None):
    """Every package under `src/robokit` imports without the official `smplx`."""
    import robokit.smplx  # noqa: F401

    # `body_lbs_torch` is lazy-exported and only resolves when `torch` is
    # installed; skip it here so the smplx-runtime guard stays orthogonal to
    # the torch-runtime guard in `tests/test_no_torch_runtime.py`.
    from robokit.helpers.humanoid_retarget.loaders import (  # noqa: F401
        SmplxMotion,
        get_smplx_motion,
        load_smplx_file,
    )
    from robokit.smplx import (  # noqa: F401
        SMPLX_JOINT_NAMES,
        SMPLX_NUM_JOINTS,
        SMPLX_STATIC_LANDMARK_NAMES,
        SMPLX_STATIC_LANDMARK_VERTEX_IDS,
        BodyModelSpec,
        BodyModelSpecTensors,
        BodyModelState,
        body_fk_warp,
        body_lbs_warp,
        from_amass_dict,
        load_smplx,
    )


def test_no_smplx_string_in_repo_sources():
    """Static check: no `import smplx`/`from smplx` under `src/` or `examples/`."""
    repo_root = Path(__file__).resolve().parent.parent
    forbidden_re = re.compile(r"^\s*(?:from\s+smplx|import\s+smplx)\b", re.MULTILINE)
    scan_dirs = [repo_root / "src" / "robokit", repo_root / "examples"]
    offenders: List[str] = []
    for scan_dir in scan_dirs:
        for py_file in scan_dir.rglob("*.py"):
            text = py_file.read_text()
            if forbidden_re.search(text):
                offenders.append(str(py_file.relative_to(repo_root)))
    assert not offenders, "official `smplx` import found:\n  " + "\n  ".join(offenders)
