"""Robot specifications and solver settings for hand retargeting."""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

from robokit.opt.multi_seed_solver import MultiSeedSolverConfig, StageConfig
from robokit.opt.sparse_lm_optimizer import SparseLMOptimizerConfig
from robokit.utils.hand_coord_utils import HandCoordinateSpec


_Correspondence = Tuple[str, str, str, str]


def _online_solver() -> MultiSeedSolverConfig:
    return MultiSeedSolverConfig(
        stages=[
            StageConfig(num_seeds=16, iters=6, lm_lambda=1e-1),
            StageConfig(num_seeds=4, iters=10, lm_lambda=1e-2),
            StageConfig(num_seeds=1, iters=0, lm_lambda=1e-2),
        ],
        cuda_graph_mode="full",
    )


def _offline_solver() -> SparseLMOptimizerConfig:
    return SparseLMOptimizerConfig(
        lm_lambda=0.5,
        max_iter=25,
        lambda_factor=2.0,
        lambda_min=1e-6,
        lambda_max=1e6,
        use_cuda_graph=True,
    )


@dataclass(frozen=True)
class HandSpec:
    """Named input topology and reference joint states for one robot hand."""

    # --- topology ---
    floating_base: bool
    """Whether retargeting optimizes the robot base transform."""
    target_names: Tuple[str, ...]
    """Ordered target names on the input point axis."""
    target_link_names: Dict[str, str]
    """Robot link name for each mapped target."""
    target_chains: Tuple[Tuple[str, ...], ...]
    """Target chains used to construct direct pairs and pinch gates."""
    root_target_name: str
    """Target that defines the input hand root."""
    contact_target_names: Tuple[str, ...]
    """Ordered targets accepted by contact solves."""

    # --- coordinates ---
    target_coord_spec: HandCoordinateSpec
    """Coordinate convention of the input root orientation."""
    root_link_coord_spec: HandCoordinateSpec
    """Coordinate convention of the robot root link."""

    # --- reference state ---
    init_q_by_name: Dict[str, float] = field(default_factory=dict)
    """Initial joint positions keyed by actuated joint name."""
    rest_q_by_name: Dict[str, float] = field(default_factory=dict)
    """Rest joint positions keyed by actuated joint name."""


@dataclass
class HandRetargetingOnlineConfig:
    """Frame-level hand-retargeting configuration."""

    # --- targets ---
    position_weights: Dict[str, float] = field(default_factory=dict)
    """Position weights keyed by target name."""
    vector_weights: Dict[_Correspondence, float] = field(default_factory=dict)
    """Vector weights keyed by `(origin_link, task_link, origin_target, task_target)`."""
    direction_weights: Dict[_Correspondence, float] = field(default_factory=dict)
    """Direction weights keyed by `(origin_link, task_link, origin_target, task_target)`."""
    pinch_correspondences: Tuple[_Correspondence, ...] = ()
    """Vector correspondences using pinch hysteresis."""

    # --- target processing ---
    position_huber_delta: float = 0.02
    """Position Huber-loss transition distance."""
    vector_target_scale: float = 1.0
    """Common scale applied to input vectors."""
    vector_robot_scale: float = 1.0
    """Common scale applied to robot vectors."""
    vector_huber_delta: Optional[float] = None
    """Optional vector Huber-loss transition distance."""
    vector_soft_gate_start_distance: Optional[float] = None
    """Target distance where soft gating starts."""
    vector_soft_gate_full_distance: Optional[float] = None
    """Target distance where soft gating reaches full weight."""
    vector_target_ema_alpha: float = 1.0
    """Input-vector exponential smoothing factor."""
    direction_target_ema_alpha: float = 1.0
    """Input-direction exponential smoothing factor."""
    output_ema_alpha: float = 1.0
    """Output-joint exponential smoothing factor."""

    # --- sampling ---
    seed: Optional[int] = None
    """Optional sampling seed."""
    sampling_distance: float = 0.2
    """Joint-state sampling distance."""
    base_sampling_distance: float = 0.1
    """Base-state sampling distance."""
    base_sampling_translation_mask: Tuple[float, float, float] = (1.0, 1.0, 0.0)
    """Base translation axes sampled during initialization."""

    # --- solver ---
    solver: MultiSeedSolverConfig = field(default_factory=_online_solver)
    """Multi-seed solver settings."""

    # --- pinch ---
    pinch_threshold: Optional[float] = None
    """Target distance that latches a pinch correspondence."""
    pinch_release_threshold: Optional[float] = None
    """Target distance that releases a pinch correspondence."""
    pinch_target_norm: float = 1e-4
    """Target vector norm while a pinch is latched."""
    gate_direction_on_pinch: bool = False
    """Whether a latched pinch disables directions on the same target chain."""

    # --- regularization ---
    q_smoothness_weight: float = 0.0
    """Joint-state temporal smoothness weight."""
    base_smoothness_weight: Optional[float] = None
    """Optional base-state temporal smoothness weight."""
    regularization_weight: float = 0.0
    """Initial-state regularization weight."""
    limit_weight: float = 1.0
    """Joint-limit weight."""
    limit_residual_mode: Literal["abs", "sqrt_abs", "exp_barrier"] = "abs"
    """Joint-limit residual transform."""

    # --- collision ---
    self_collision_weight: float = 0.0
    """Self-collision weight."""
    self_collision_margin: float = 0.005
    """Self-collision clearance margin."""
    self_collision_max_active_pairs: int = 50
    """Maximum active self-collision pairs."""


@dataclass
class HandRetargetingOfflineConfig:
    """Trajectory, contact, and anchored retargeting configuration."""

    # --- targets ---
    target_scale: float = 1.0
    """Common scale applied to input points around the root."""
    global_position_weight: float = 20.0
    """Named link-position tracking weight."""
    root_position_weight: float = 1.0
    """Root-position tracking weight."""
    root_orientation_weight: float = 0.2
    """Root-orientation tracking weight."""
    vector_weight: float = 0.5
    """Kinematic-pair vector weight."""
    direction_weight: float = 10.0
    """Kinematic-pair direction weight."""
    pair_mode: Literal["direct", "all"] = "direct"
    """Whether to use direct or all selected-link pairs."""
    include_root_target: bool = True
    """Whether pair construction includes root connections."""
    # --- smoothness ---
    velocity_weight: float = 9.0
    """Joint-velocity smoothness weight."""
    acceleration_weight: float = 2.0
    """Joint-acceleration smoothness weight."""
    root_position_velocity_weight: float = 10.0
    """Base translation-velocity weight."""
    root_orientation_velocity_weight: float = 5.0
    """Base angular-velocity weight."""

    # --- regularization ---
    limit_weight: float = 100.0
    """Joint-limit weight."""
    rest_weight: float = 0.05
    """Rest-state regularization weight."""

    # --- contact and collision ---
    contact_weight: float = 100.0
    """Contact-position weight."""
    contact_margin: float = 0.01
    """Contact clearance margin."""
    collision_weight: float = 20.0
    """Scene-collision weight."""
    collision_margin: float = 0.005
    """Scene-collision clearance margin."""

    # --- anchored refinement ---
    anchor_q_weight: float = 1.0
    """Anchored joint-state weight."""
    anchor_base_weight: float = 20.0
    """Anchored base-state weight."""
    optimize_base: bool = True
    """Whether anchored refinement may move the base."""
    active_joint_names: Optional[Tuple[str, ...]] = None
    """Optional joints optimized during anchored refinement."""
    locked_prefix_frames: int = 0
    """Leading anchored frames held fixed."""

    # --- solver ---
    solver: SparseLMOptimizerConfig = field(default_factory=_offline_solver)
    """Sparse trajectory solver settings."""


__all__ = ["HandRetargetingOfflineConfig", "HandRetargetingOnlineConfig", "HandSpec"]
