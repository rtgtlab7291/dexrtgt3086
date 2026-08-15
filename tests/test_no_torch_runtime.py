"""Regression guard: the runtime path must never import `torch`.

Main's #32 made `torch` an optional runtime extra; `robokit.smplx`
(engine + SMPL-X + MANO loaders) keeps that invariant by using
numpy-backed `BodyModelState` and Warp kernels. The torch path
(`body_lbs_torch`, `mano_forward_pca_mm`) is still available but
lazy-exported via PEP 562 - only callers that explicitly request it
pull `torch` into `sys.modules`.
"""

import re
import sys
from pathlib import Path
from typing import List

import pytest


class _BlockTorch:
    """`meta_path` finder that turns `import torch` into `ImportError`."""

    def find_module(self, name, path=None):
        if name == "torch" or name.startswith("torch."):
            return self
        return None

    def load_module(self, name):
        raise ImportError(
            f"Runtime guard: {name!r} import is forbidden on the runtime path. "
            f"Install via `uv sync --extra pt-cu128` for the oracle / differentiable tooling."
        )


@pytest.fixture
def block_torch(monkeypatch: pytest.MonkeyPatch):
    for mod_name in list(sys.modules.keys()):
        if mod_name == "torch" or mod_name.startswith("torch."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockTorch(), *sys.meta_path])


def test_smplx_runtime_imports(block_torch: None):
    """Importing `robokit.smplx` (engine + SMPL-X + MANO) works without `torch`."""
    import robokit.smplx  # noqa: F401
    from robokit.helpers.hand_retargeting.dexycb_utils import compute_world_hand_geometry  # noqa: F401
    from robokit.smplx import (  # noqa: F401
        MANO_JOINT_NAMES,
        SMPLX_JOINT_NAMES,
        BodyModelSpec,
        BodyModelSpecTensors,
        BodyModelState,
        body_fk_warp,
        body_lbs_warp,
        from_amass_dict,
        from_dexycb_pose,
        load_mano,
        load_smplx,
    )


def test_humanoid_smplx_loader_imports(block_torch: None):
    """Humanoid-retarget runtime loader imports without `torch`."""
    from robokit.helpers.humanoid_retarget.loaders import (  # noqa: F401
        SmplxMotion,
        get_smplx_motion,
        load_smplx_file,
    )


def test_no_torch_string_in_runtime_sources():
    """Static check: no top-level `import torch`/`from torch` in the numpy+warp packages.

    Opt-in torch modules (`smplx/torch_lbs.py`, `smplx/mano_torch_forward.py`)
    are exempt - they are lazy-exported via PEP 562 `__getattr__`.
    """
    repo_root = Path(__file__).resolve().parent.parent
    # Column-0 only - TYPE_CHECKING-guarded torch imports are indented and don't
    # execute at runtime, so they are allowed (see `smplx/state.py`).
    forbidden_re = re.compile(r"^(?:from\s+torch|import\s+torch)\b", re.MULTILINE)
    runtime_files = [
        repo_root / "src" / "robokit" / "smplx" / "__init__.py",
        repo_root / "src" / "robokit" / "smplx" / "amass.py",
        repo_root / "src" / "robokit" / "smplx" / "dexycb_state.py",
        repo_root / "src" / "robokit" / "smplx" / "mano_constants.py",
        repo_root / "src" / "robokit" / "smplx" / "mano_loader.py",
        repo_root / "src" / "robokit" / "smplx" / "smplx_constants.py",
        repo_root / "src" / "robokit" / "smplx" / "smplx_loader.py",
        repo_root / "src" / "robokit" / "smplx" / "spec.py",
        repo_root / "src" / "robokit" / "smplx" / "spec_tensors.py",
        repo_root / "src" / "robokit" / "smplx" / "state.py",
        repo_root / "src" / "robokit" / "smplx" / "warp_lbs.py",
        repo_root / "src" / "robokit" / "helpers" / "humanoid_retarget" / "loaders" / "smplx.py",
        repo_root / "src" / "robokit" / "helpers" / "hand_retargeting" / "dexycb_utils.py",
    ]
    offenders: List[str] = []
    for py_file in runtime_files:
        assert py_file.exists(), f"expected runtime file missing: {py_file}"
        text = py_file.read_text()
        if forbidden_re.search(text):
            offenders.append(str(py_file.relative_to(repo_root)))
    assert not offenders, "top-level torch import on runtime path:\n  " + "\n  ".join(offenders)
