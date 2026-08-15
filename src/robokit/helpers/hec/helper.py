# pyright: reportMissingImports=false
"""Hand-eye camera extrinsic calibration: mask/depth alignment via coarse-to-fine multi-seed LM."""

from typing import List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import warp as wp
from jaxtyping import Float, Int

from robokit.helpers.hec.config import HECHelperConfig
from robokit.lie.se3 import SE3Var, se3_compose, se3_exp, se3_from_matrix, se3_to_matrix
from robokit.opt.multi_seed_solver import MultiSeedSolver, MultiSeedSolverConfig, StageConfig
from robokit.opt.var_values import VarValues
from robokit.robo import Robot
from robokit.terms.dense.mask_alignment_task import (
    MaskAlignmentNormalEqTask,
    _precompute_merged_mesh,
    _render_batch,
    dr,
)
from robokit.terms.task import ResidualTask
from robokit.utils.warp_utils import wp_vec6, wp_vec7


def build_robot_render_data(
    robot: Robot,
    q: Float[torch.Tensor, "n_obs num_dofs"],
    T_mount_base: Optional[Float[torch.Tensor, "n_obs 4 4"]] = None,
    link_names: Optional[List[str]] = None,
) -> Tuple[
    Float[torch.Tensor, "n_obs num_links 4 4"],
    List[Float[torch.Tensor, "num_vertices 3"]],
    List[Int[torch.Tensor, "num_faces 3"]],
]:
    """Build render inputs (FK link poses + GPU mesh buffers) for a robot.

    Args:
        robot: Robot loaded with `load_meshes=True`.
        q: Joint positions `[N, num_dofs]`.
        T_mount_base: Per-sample mount pose `[N, 4, 4]`. When given, link poses
            are `T_mount_link` instead of `T_base_link` (mounted camera: the
            extrinsic is then solved in the mount frame).
        link_names: Links to render; `None` selects all links with visual meshes.

    Returns:
        Tuple of `(link_poses [N, L, 4, 4], link_vertices, link_faces)` over
        the `L` selected links.

    """
    mesh_dict = robot.spec.get_link_meshes(mode="visual")
    if link_names is None:
        names = [n for n in robot.spec.link_names if n in mesh_dict]
    else:
        missing = [n for n in link_names if n not in mesh_dict]
        if missing:
            raise ValueError(f"links without visual meshes: {missing}")
        names = list(link_names)
    if not names:
        raise ValueError("robot has no visual meshes; load it with load_meshes=True")
    device = q.device
    link_vertices: List[torch.Tensor] = []
    link_faces: List[torch.Tensor] = []
    for n in names:
        mesh = cast(trimesh.Trimesh, mesh_dict[n])
        link_vertices.append(torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)).to(device))
        link_faces.append(torch.from_numpy(np.asarray(mesh.faces, dtype=np.int32)).to(device))
    idx = [robot.spec.link_names.index(n) for n in names]
    link_poses = robot.forward_kinematics_via_matrix_torch(q)[:, idx]
    if T_mount_base is not None:
        link_poses = T_mount_base.to(link_poses)[:, None] @ link_poses
    return link_poses, link_vertices, link_faces


def compute_robot_masks(
    robot: Robot,
    q: Float[torch.Tensor, "n_obs num_dofs"],
    camera_intrinsic: Float[torch.Tensor, "3 3"],
    extrinsic: Float[torch.Tensor, "4 4"],
    height: int,
    width: int,
    T_mount_base: Optional[Float[torch.Tensor, "n_obs 4 4"]] = None,
    link_names: Optional[List[str]] = None,
) -> Float[torch.Tensor, "n_obs height width"]:
    """Render robot silhouette masks (visualization; same renderer as the cost term).

    Args:
        robot: Robot loaded with `load_meshes=True`.
        q: Joint positions `[N, num_dofs]`.
        camera_intrinsic: Camera intrinsic `[3, 3]`.
        extrinsic: Camera extrinsic `[4, 4]` (`T_camera_base`, or
            `T_camera_mount` with `T_mount_base`).
        height: Image height in pixels.
        width: Image width in pixels.
        T_mount_base: Per-sample mount pose `[N, 4, 4]` (mounted camera).
        link_names: Links to render; `None` selects all links with visual meshes.

    Returns:
        Anti-aliased silhouette masks `[N, H, W]` in `[0, 1]`.
    """
    link_poses, link_vertices, link_faces = build_robot_render_data(robot, q, T_mount_base, link_names)
    merged_faces, v_base = _precompute_merged_mesh(link_vertices, link_faces, link_poses)
    with torch.no_grad():
        masks, _ = _render_batch(
            dr.RasterizeCudaContext(), extrinsic, camera_intrinsic, merged_faces, v_base, height, width, False
        )
    return masks


class HECHelper:
    """Solve the camera extrinsic from robot silhouette masks and/or depth maps.

    Lifecycle:
        hec = HECHelper(config, robot, camera_intrinsic, height, width)
        T = hec.solve(initial, target_masks=masks, q=q)  # binds data, builds solver
        T = hec.solve(other_initial)                     # re-solves on bound data
    """

    def __init__(
        self,
        config: HECHelperConfig,
        robot: Robot,
        camera_intrinsic: Float[torch.Tensor, "3 3"],
        height: int,
        width: int,
        link_names: Optional[List[str]] = None,
    ):
        wp.init()
        self.camera_intrinsic = camera_intrinsic
        self.height = height
        self.width = width
        self.config = config
        self.robot = robot
        self.link_names = link_names
        self._wp_device = "cuda:0"
        self._initialized = False

    def _build(
        self,
        target_masks: Optional[Float[torch.Tensor, "n_obs height width"]],
        target_depths: Optional[Float[torch.Tensor, "n_obs height width"]],
        q: Float[torch.Tensor, "n_obs num_dofs"],
        T_mount_base: Optional[Float[torch.Tensor, "n_obs 4 4"]],
    ):
        """Bind observation data and build per-level tasks + the multi-seed solver."""
        link_poses, self.link_vertices, self.link_faces = build_robot_render_data(
            self.robot, q, T_mount_base, self.link_names
        )
        self.target_masks = target_masks
        self.target_depths = target_depths
        self.link_poses = link_poses
        levels = list(self.config.levels)
        stage_terms: List[List[ResidualTask]] = []
        stages: List[StageConfig] = []
        self._mask_tasks: List[MaskAlignmentNormalEqTask] = []

        # --- rank observations ---
        # rank by robot-mask area (= information content); fine stages slice the top-K
        if self.target_masks is not None:
            areas = self.target_masks.sum(dim=(1, 2))
        else:
            assert self.target_depths is not None
            areas = (self.target_depths > 0).sum(dim=(1, 2)).float()
        N = areas.shape[0]
        area_order = torch.argsort(areas, descending=True)

        # --- per-level tasks ---
        for level in levels:
            if level.use_depth and self.target_depths is None:
                raise ValueError("level has use_depth but no target_depths were given")
            if level.obs_subset_size is not None and level.obs_subset_size < N:
                idx = area_order[: level.obs_subset_size]
                masks_full = self.target_masks[idx].contiguous() if self.target_masks is not None else None
                depths_full = self.target_depths[idx].contiguous() if self.target_depths is not None else None
                link_poses_l = self.link_poses[idx].contiguous()
            else:
                masks_full = self.target_masks
                depths_full = self.target_depths
                link_poses_l = self.link_poses

            # cap downscale to keep >=32 px on the short side (blurred cost too flat below to rank seeds)
            ds = min(level.downscale, max(1, min(self.height, self.width) // 32))
            if ds == 1:
                H_l, W_l = self.height, self.width
                K_l = self.camera_intrinsic
                masks_l = masks_full
                depths_l = depths_full
            else:
                H_l = self.height // ds
                W_l = self.width // ds
                K_l = self.camera_intrinsic.clone()
                K_l[:2] /= ds
                masks_l = F.avg_pool2d(masks_full.unsqueeze(1), ds).squeeze(1) if masks_full is not None else None
                # stride subsample at avg_pool pixel centers, not avg_pool (averaging bleeds hole-zeros into depths)
                depths_l = (
                    depths_full[:, ds // 2 : H_l * ds : ds, ds // 2 : W_l * ds : ds].contiguous()
                    if depths_full is not None
                    else None
                )
            task = MaskAlignmentNormalEqTask(
                camera_intrinsic=K_l,
                target_masks=masks_l,
                link_poses=link_poses_l,
                link_vertices=self.link_vertices,
                link_faces=self.link_faces,
                height=H_l,
                width=W_l,
                blur_sigma=level.blur_sigma,
                patience=level.patience,
                target_depths=depths_l if level.use_depth else None,
                depth_weight=self.config.depth_weight,
                depth_huber_delta=self.config.depth_huber_delta,
            )
            stage_terms.append([task])
            self._mask_tasks.append(task)
            stages.append(
                StageConfig(
                    num_seeds=level.num_seeds,
                    iters=level.max_iter,
                    lm_lambda=self.config.lm_lambda,
                    early_stopping_interval=level.early_stopping_interval,
                )
            )

        # --- solver ---
        first_num_seeds = stages[0].num_seeds
        init_v7 = np.tile(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32), (first_num_seeds, 1))
        self._initial_var = VarValues(robot=SE3Var(wp.from_numpy(init_v7, dtype=wp_vec7, device=self._wp_device)))

        solver_config = MultiSeedSolverConfig(
            stages=stages,
            cuda_graph_mode=self.config.cuda_graph_mode,
            lambda_factor=self.config.lambda_factor,
            rho_min=self.config.rho_min,
        )
        self._solver = MultiSeedSolver(terms=stage_terms, config=solver_config, device=self._wp_device)
        self._solver.setup(self._initial_var)
        self._rng = np.random.default_rng(self.config.seed)
        self._initialized = True

    def solve(
        self,
        initial_extrinsic: Float[torch.Tensor, "4 4"],
        target_masks: Optional[Float[torch.Tensor, "n_obs height width"]] = None,
        target_depths: Optional[Float[torch.Tensor, "n_obs height width"]] = None,
        q: Optional[Float[torch.Tensor, "n_obs num_dofs"]] = None,
        T_mount_base: Optional[Float[torch.Tensor, "n_obs 4 4"]] = None,
    ) -> Float[torch.Tensor, "4 4"]:
        """Solve the camera extrinsic starting from `initial_extrinsic`.

        Passing any target (re)binds data and rebuilds the solver; omitting all targets re-solves on bound data.

        Args:
            initial_extrinsic: Initial guess `[4, 4]` (OpenCV `T_camera_base`,
                or `T_camera_mount` with a mounted camera).
            target_masks: Robot silhouette masks `[N, H, W]` in `[0, 1]`.
            target_depths: Robot-only depth maps `[N, H, W]` in meters, 0 = invalid.
            q: Joint positions `[N, num_dofs]`; required with target data.
            T_mount_base: Per-sample mount pose `[N, 4, 4]` (mounted camera).

        Returns:
            Optimized extrinsic `[4, 4]` on `initial_extrinsic`'s device.

        """
        if target_masks is not None or target_depths is not None:
            if q is None:
                raise ValueError("pass q [N, num_dofs] with target data")
            self._build(target_masks, target_depths, q, T_mount_base)
        elif not self._initialized:
            raise ValueError("no data bound: pass target_masks/target_depths with q")
        dev = initial_extrinsic.device
        first_num_seeds = self.config.levels[0].num_seeds

        # --- reset per-task early-stopping state from a prior solve ---
        for task in self._mask_tasks:
            task._best_cost = float("inf")
            task._last_improvement = 0
            task._best_xyz_wxyz = None

        # --- sample seeds around the initial guess ---
        mat_np = initial_extrinsic.detach().cpu().numpy().astype(np.float32)
        T_init = se3_from_matrix(wp.from_numpy(mat_np.reshape(1, 4, 4), dtype=wp.mat44, device=self._wp_device))
        init_v7 = T_init.numpy()  # [1, 7]

        seeds_np = np.tile(init_v7, (first_num_seeds, 1)).astype(np.float32)
        twists_np = np.zeros((first_num_seeds, 6), dtype=np.float32)
        if first_num_seeds > 1:
            # uniform-shell sampling: random direction, magnitude in [0.5, 1.0] * noise (Gaussian clusters near zero)
            n = first_num_seeds - 1
            t_dirs = self._rng.standard_normal((n, 3)).astype(np.float32)
            t_dirs /= np.linalg.norm(t_dirs, axis=1, keepdims=True) + 1e-12
            t_mags = self._rng.uniform(0.5, 1.0, size=(n, 1)).astype(np.float32) * self.config.seed_noise_t
            r_dirs = self._rng.standard_normal((n, 3)).astype(np.float32)
            r_dirs /= np.linalg.norm(r_dirs, axis=1, keepdims=True) + 1e-12
            r_mags = self._rng.uniform(0.5, 1.0, size=(n, 1)).astype(np.float32) * self.config.seed_noise_r
            twists_np[1:, :3] = t_dirs * t_mags
            twists_np[1:, 3:] = r_dirs * r_mags
        # seed_noise_t/r displace the camera itself, so compose on the left: right multiplication
        # would read the shell in the base (or mount) frame, where a 0.7 rad twist swings the
        # camera far off target
        se3_compose(
            se3_exp(wp.from_numpy(twists_np, dtype=wp_vec6, device=self._wp_device)),
            wp.from_numpy(seeds_np, dtype=wp_vec7, device=self._wp_device),
            out=cast(SE3Var, self._initial_var.get("robot")).xyz_wxyz,
        )

        # --- run ---
        best_var, best_costs = self._solver.solve(self._initial_var)
        best_state = cast(SE3Var, best_var.get("robot"))
        best_mat = se3_to_matrix(best_state.xyz_wxyz).numpy()[0]

        self.last_final_cost = float(best_costs.numpy()[0])

        return torch.as_tensor(best_mat, device=dev)
