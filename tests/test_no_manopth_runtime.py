"""Regression guard: the runtime path must never import `manopth` or `chumpy`."""

import os
import sys
from pathlib import Path
from typing import List

import pytest


_FORBIDDEN = ("manopth", "chumpy")


class _BlockManoLegacy:
    """meta_path finder that turns `import manopth`/`chumpy` into ImportError."""

    def find_module(self, name, path=None):
        if name in _FORBIDDEN or any(name.startswith(p + ".") for p in _FORBIDDEN):
            return self
        return None

    def load_module(self, name):
        raise ImportError(
            f"Runtime guard: {name!r} import is forbidden. "
            f"`manopth`/`chumpy` are no longer supported anywhere in this repo; use `robokit.smplx` instead."
        )


@pytest.fixture
def block_mano_legacy(monkeypatch: pytest.MonkeyPatch):
    # Drop any cached imports of the forbidden modules so the blocker actually fires.
    for mod_name in list(sys.modules.keys()):
        if mod_name in _FORBIDDEN or any(mod_name.startswith(p + ".") for p in _FORBIDDEN):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockManoLegacy(), *sys.meta_path])


def test_robokit_runtime_imports(block_mano_legacy: None):
    """Every package on the runtime path imports without `manopth`/`chumpy`."""
    # body-model package (engine + SMPL-X + MANO format adapters)
    # DexYCB utilities (the historical home of the lazy `manopth` import)
    import robokit.helpers.hand_retargeting.dexycb_utils  # noqa: F401
    import robokit.smplx  # noqa: F401
    from robokit.helpers.hand_retargeting.dexycb_utils import (  # noqa: F401
        DexYCBVideoDataset,
        compute_world_hand_geometry,
        load_dexycb_keypoints,
    )

    # humanoid retarget loader
    from robokit.helpers.humanoid_retarget.loaders import (  # noqa: F401
        SmplxMotion,
        get_smplx_motion,
        load_smplx_file,
    )
    from robokit.smplx import (  # noqa: F401
        body_fk_warp,
        body_lbs_warp,
        from_amass_dict,
        from_dexycb_pose,
        load_mano,
        load_smplx,
    )


@pytest.mark.skipif(
    bool(os.environ.get("GITHUB_ACTIONS")),
    reason="skip in CI: importing the examples triggers HF download-on-import (assets.motions)",
)
def test_examples_hand_retarget_imports_clean(block_mano_legacy: None):
    """Every `examples/hand_retarget/*.py` imports without `manopth`/`chumpy`.

    Skip the whole suite when torch is unavailable so this stays a pure
    `manopth` guard, orthogonal to `tests/test_no_torch_runtime.py`.
    """
    pytest.importorskip("torch", reason="examples import torch; install via `--extra pt-*`")
    import importlib.util

    repo_root = Path(__file__).resolve().parent.parent
    hand_retarget_dir = repo_root / "examples" / "hand_retarget"
    example_files: List[Path] = sorted(hand_retarget_dir.glob("*.py"))
    assert example_files, f"no hand examples found under {hand_retarget_dir}"

    for example_path in example_files:
        spec = importlib.util.spec_from_file_location(example_path.stem, example_path)
        assert spec is not None and spec.loader is not None, f"could not spec-load {example_path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def test_no_manopth_string_in_runtime_sources():
    """Static check: no `import manopth`/`from manopth` on the runtime path."""
    import re

    repo_root = Path(__file__).resolve().parent.parent
    forbidden_re = re.compile(r"^\s*(?:from\s+(manopth|chumpy)|import\s+(manopth|chumpy))\b", re.MULTILINE)
    runtime_dirs = [
        repo_root / "src" / "robokit",
        repo_root / "examples" / "hand_retarget",
        repo_root / "examples" / "humanoid_retarget",
    ]
    offenders: List[str] = []
    for runtime_dir in runtime_dirs:
        for py_file in runtime_dir.rglob("*.py"):
            text = py_file.read_text()
            if forbidden_re.search(text):
                offenders.append(str(py_file.relative_to(repo_root)))
    assert not offenders, "manopth/chumpy import found on runtime path:\n  " + "\n  ".join(offenders)
