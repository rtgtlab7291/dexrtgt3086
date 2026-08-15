"""Module-level IK preset constants.

Each preset is a fully-built robot-free `IKConfig`. Pass to
`IK(presets.basic, robot, link)` to bind a robot at construction time.
"""

from robokit.helpers.ik.config import IKConfig
from robokit.opt.multi_seed_solver import MultiSeedSolverConfig, StageConfig
from robokit.terms.dense.position_limit import PositionLimit
from robokit.terms.dense.position_task import PositionTask
from robokit.terms.dense.rotation_task import RotationTask
from robokit.terms.dense.smoothness_task import SmoothnessTask


basic = (
    IKConfig(init_sample_range=1.0)
    .add(PositionTask(weight=20.0))
    .add(RotationTask(weight=10.0))
    .add(PositionLimit(weight=50.0))
)

smooth = (
    IKConfig(init_sample_range=1.0)
    .add(PositionTask(weight=20.0))
    .add(RotationTask(weight=10.0))
    .add(PositionLimit(weight=50.0))
    .add(SmoothnessTask(weight=0.1))
)

single_seed = (
    IKConfig(
        solver=MultiSeedSolverConfig(
            stages=[StageConfig(num_seeds=1, iters=30, lm_lambda=1.0)],
            cuda_graph_mode="full",
        ),
    )
    .add(PositionTask(weight=20.0))
    .add(RotationTask(weight=10.0))
    .add(PositionLimit(weight=50.0))
    .add(SmoothnessTask(weight=0.1))
)


__all__ = ["basic", "single_seed", "smooth"]
