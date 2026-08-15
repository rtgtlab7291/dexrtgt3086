"""robokit.assets: fetch, per-asset modules (download-on-import), link."""

import os

import pytest
from huggingface_hub import get_token

from robokit.assets import HF_REPO_DIR, fetch


_G1_SPHERES = "robots/collision_spheres/g1/collision_spheres.yaml"
# TODO(ci): make CI-friendly. These fetch assets from HF, which rate-limits (429)
# in CI. Vendor small fixtures (or mark network + warm the cache) then re-enable.
_RUNNABLE = get_token() is not None and not os.environ.get("GITHUB_ACTIONS")


class TestLink:
    def test_link_to_hf_repo_is_idempotent(self, tmp_path):
        if not HF_REPO_DIR.is_dir():
            pytest.skip("hf_repo clone absent; link() would git-clone")
        from robokit.assets import link

        target = tmp_path / "assets"
        assert link(target=target) == target
        assert target.is_symlink()
        assert target.resolve() == HF_REPO_DIR.resolve()
        # second call is a no-op, not an error
        assert link(target=target) == target


@pytest.mark.skipif(not _RUNNABLE, reason="needs HuggingFace auth (set HF_TOKEN)")
class TestFetch:
    def test_fetch_returns_snapshot_with_requested_file(self):
        snapshot = fetch([_G1_SPHERES.rsplit("/", 1)[0] + "/**"])
        assert (snapshot / _G1_SPHERES).is_file()

    def test_asset_module_paths_resolve(self):
        # Importing an asset module downloads it (download-on-import) and exposes resolved paths.
        from robokit.assets.robots.humanoids import unitree_g1

        assert unitree_g1.URDF_PATH.is_file()
        assert unitree_g1.COLLISION_SPHERE_PATH.is_file()

    def test_new_assembly_modules_resolve(self):
        # Newly added assembly / end-effector modules download-on-import and resolve.
        from robokit.assets.robots.arms import xarm6_sharpa
        from robokit.assets.robots.hands import mano_hand

        assert mano_hand.URDF_PATH.is_file()
        assert mano_hand.COLLISION_SPHERE_PATH.is_file()
        assert xarm6_sharpa.URDF_PATH.is_file()
        assert xarm6_sharpa.COLLISION_SPHERE_PATH.is_file()
