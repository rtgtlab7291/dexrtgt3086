"""End-to-end HECHelper tests (rgb/rgbd/depth) on a toy two-box robot."""

import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation


pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch")
mask_task = pytest.importorskip("robokit.terms.dense.mask_alignment_task")
hec_module = pytest.importorskip("robokit.helpers.hec")
hec_config = pytest.importorskip("robokit.helpers.hec.config")

_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="nvdiffrast requires CUDA")

_H = _W = 128
_N_OBS = 4

_URDF = """<robot name="toy">
  <link name="base">
    <visual><origin xyz="-0.25 0 0"/><geometry><mesh filename="box0.obj"/></geometry></visual>
  </link>
  <link name="arm">
    <visual><geometry><mesh filename="box1.obj"/></geometry></visual>
  </link>
  <joint name="j0" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0.25 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _toy_scene(tmp_dir, device: torch.device) -> dict:
    """Two-box single-joint robot rendered from a known extrinsic."""
    import trimesh

    from robokit.robo import Robot

    trimesh.creation.box(extents=(0.4, 0.25, 0.3)).export(tmp_dir / "box0.obj")
    trimesh.creation.box(extents=(0.15, 0.5, 0.2)).export(tmp_dir / "box1.obj")
    (tmp_dir / "toy.urdf").write_text(_URDF)
    robot = Robot.load(str(tmp_dir / "toy.urdf"), load_meshes=True)

    rng = np.random.default_rng(0)
    q = torch.tensor(rng.uniform(-0.6, 0.6, (_N_OBS, 1)), dtype=torch.float32, device=device)

    gt_extrinsic = torch.eye(4, dtype=torch.float32, device=device)
    gt_extrinsic[2, 3] = 1.5  # boxes ~1.5 m in front of the camera
    intrinsic = torch.tensor(
        [[150.0, 0.0, _W / 2], [0.0, 150.0, _H / 2], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device
    )

    link_poses, link_vertices, link_faces = hec_module.build_robot_render_data(robot, q)
    merged_faces, v_base = mask_task._precompute_merged_mesh(link_vertices, link_faces, link_poses)
    glctx = mask_task.dr.RasterizeCudaContext()
    with torch.no_grad():
        masks, positions = mask_task._render_batch(glctx, gt_extrinsic, intrinsic, merged_faces, v_base, _H, _W)
    depths = positions[..., 2].clamp(min=0.0) * (masks > 0.5)

    return dict(
        robot=robot,
        q=q,
        intrinsic=intrinsic,
        gt_extrinsic=gt_extrinsic,
        masks=masks.contiguous(),
        depths=depths.contiguous(),
    )


def _perturbed(gt: torch.Tensor) -> torch.Tensor:
    T = gt.clone()
    R = torch.from_numpy(Rotation.from_euler("xyz", [0.05, -0.06, 0.04]).as_matrix().astype(np.float32))
    T[:3, :3] = R.to(gt.device) @ T[:3, :3]
    T[:3, 3] += torch.tensor([0.03, -0.02, 0.03], device=gt.device)
    return T


def _pose_error(predicted: torch.Tensor, gt: torch.Tensor) -> tuple:
    t_err = (gt[:3, 3] - predicted[:3, 3]).norm().item()
    R_rel = gt[:3, :3] @ predicted[:3, :3].t()
    r_err = torch.acos(((R_rel.trace() - 1) / 2).clamp(-1, 1)).item() * 180 / np.pi
    return t_err, r_err


def _solve(scene: dict, masks, depths, levels) -> tuple:
    helper = hec_module.HECHelper(
        camera_intrinsic=scene["intrinsic"],
        height=_H,
        width=_W,
        robot=scene["robot"],
        config=hec_config.HECHelperConfig(levels=levels),
    )
    predicted = helper.solve(
        _perturbed(scene["gt_extrinsic"]),
        target_masks=masks,
        target_depths=depths,
        q=scene["q"],
    )
    return _pose_error(predicted, scene["gt_extrinsic"])


@_REQUIRES_CUDA
class TestHECHelperDepth:
    @pytest.fixture(scope="class")
    def scene(self, tmp_path_factory):
        wp.init()
        try:
            return _toy_scene(tmp_path_factory.mktemp("toy_robot"), torch.device("cuda"))
        except (ImportError, RuntimeError) as e:  # nvdiffrast plugin/torch mismatch
            pytest.skip(f"nvdiffrast unavailable: {e}")

    def test_depth_only_convergence(self, scene):
        # masks=None also exercises the valid-depth-pixel observation ranking
        # (obs_subset_size < N)
        levels = [
            hec_config.PyramidLevel(
                downscale=4, num_seeds=4, max_iter=20, blur_sigma=0.0, patience=5, use_depth=True, obs_subset_size=3
            ),
            hec_config.PyramidLevel(downscale=1, num_seeds=1, max_iter=40, blur_sigma=0.0, patience=8, use_depth=True),
        ]
        t_err, r_err = _solve(scene, None, scene["depths"], levels)
        assert t_err < 0.01 and r_err < 1.0, f"depth-only: {t_err * 100:.2f}cm / {r_err:.2f}deg"

    def test_rgbd_convergence(self, scene):
        levels = [
            hec_config.PyramidLevel(downscale=4, num_seeds=4, max_iter=20, blur_sigma=1.2, patience=5),
            hec_config.PyramidLevel(downscale=1, num_seeds=1, max_iter=40, blur_sigma=0.0, patience=8, use_depth=True),
        ]
        t_err, r_err = _solve(scene, scene["masks"], scene["depths"], levels)
        assert t_err < 0.01 and r_err < 1.0, f"rgbd: {t_err * 100:.2f}cm / {r_err:.2f}deg"

    def test_requires_some_target(self, scene):
        with pytest.raises(ValueError):
            _solve(scene, None, None, list(hec_config._default_levels()))

    def test_use_depth_without_depths_raises(self, scene):
        levels = [
            hec_config.PyramidLevel(downscale=1, num_seeds=1, max_iter=5, blur_sigma=0.0, patience=2, use_depth=True)
        ]
        with pytest.raises(ValueError):
            _solve(scene, scene["masks"], None, levels)
