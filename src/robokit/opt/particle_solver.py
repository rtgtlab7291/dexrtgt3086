# pyright: reportArgumentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportOperatorIssue=false
from dataclasses import dataclass
from typing import Optional, Union, cast

import numpy as np
import warp as wp

from robokit.opt.optimizer import aggregate_residuals_to_costs_kernel
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig
from robokit.opt.var_values import VarValues
from robokit.terms.task import SparseTask


__all__ = ["ParticleSolverConfig", "ParticleSolver"]


@dataclass
class ParticleSolverConfig(PopulationSolverConfig):
    """Position-space MPPI sampling configuration."""

    num_particles: int = 100
    init_std: float = 0.1
    beta: float = 0.1
    base_seed: int = 0
    lock_dim: int = 0
    gamma: float = 0.9
    min_std: float = 0.01
    active_dof_mask: Optional[wp.array] = None
    std_scale: Optional[wp.array] = None


@wp.kernel
def _mppi_sample_kernel(
    seed: int,
    salt: int,
    num_particles: int,
    tangent_dim: int,
    lock_dim: int,
    active_dof_mask: wp.array1d(dtype=wp.float32),
    std: wp.array2d(dtype=wp.float32),
    noise: wp.array2d(dtype=wp.float32),
):
    row_idx, dof_idx = wp.tid()
    if (row_idx % num_particles) == 0 or dof_idx < lock_dim or active_dof_mask[dof_idx] == 0.0:
        noise[row_idx, dof_idx] = 0.0
    else:
        state = wp.rand_init(seed, (salt * 1000003 + row_idx) * tangent_dim + dof_idx)
        noise[row_idx, dof_idx] = std[row_idx / num_particles, dof_idx] * wp.randn(state)


@wp.kernel
def _mppi_fill_std_kernel(
    scale: wp.array1d(dtype=wp.float32),
    min_std: float,
    init_std: float,
    std: wp.array2d(dtype=wp.float32),
):
    instance_idx, dim_idx = wp.tid()
    std[instance_idx, dim_idx] = min_std + (init_std - min_std) * scale[instance_idx]


@wp.kernel
def _mppi_broadcast_add_kernel(
    mean: wp.array2d(dtype=wp.float32),
    noise: wp.array2d(dtype=wp.float32),
    particles_per_instance: int,
    out: wp.array2d(dtype=wp.float32),
):
    row_idx, dof_idx = wp.tid()
    out[row_idx, dof_idx] = mean[row_idx / particles_per_instance, dof_idx] + noise[row_idx, dof_idx]


@wp.kernel
def _mppi_add_inplace_kernel(delta: wp.array2d(dtype=wp.float32), value: wp.array2d(dtype=wp.float32)):
    batch_idx, dof_idx = wp.tid()
    value[batch_idx, dof_idx] = value[batch_idx, dof_idx] + delta[batch_idx, dof_idx]


@wp.kernel
def _mppi_cov_and_delta_kernel(
    weights: wp.array2d(dtype=wp.float32),
    noise: wp.array2d(dtype=wp.float32),
    num_particles: int,
    gamma: float,
    min_std: float,
    std: wp.array2d(dtype=wp.float32),
    delta: wp.array2d(dtype=wp.float32),
):
    batch_idx, dof_idx = wp.tid()
    start = batch_idx * num_particles
    acc = wp.float32(0.0)
    var = wp.float32(0.0)
    for p in range(num_particles):
        n = noise[start + p, dof_idx]
        w = weights[batch_idx, p]
        acc += w * n
        var += w * n * n
    delta[batch_idx, dof_idx] = acc
    std[batch_idx, dof_idx] = max(
        wp.sqrt((1.0 - gamma) * std[batch_idx, dof_idx] * std[batch_idx, dof_idx] + gamma * var), min_std
    )


@wp.kernel
def _mppi_weights_kernel(
    costs: wp.array1d(dtype=wp.float32),
    num_particles: int,
    beta: float,
    weights: wp.array2d(dtype=wp.float32),
):
    batch_idx = wp.tid()
    start = batch_idx * num_particles
    minimum = costs[start]
    for p in range(1, num_particles):
        minimum = min(minimum, costs[start + p])
    total = wp.float32(0.0)
    for p in range(num_particles):
        weight = wp.exp(-(costs[start + p] - minimum) / beta)
        weights[batch_idx, p] = weight
        total += weight
    for p in range(num_particles):
        weights[batch_idx, p] = weights[batch_idx, p] / total


class _MppiStage:
    def __init__(self, mean_var: VarValues, config: ParticleSolverConfig, residual_dim: int, device):
        self.instances = mean_var.batch_size
        self.dim = mean_var.tangent_dim
        self.num_particles = config.num_particles
        self.active_dof_mask = config.active_dof_mask
        if self.active_dof_mask is None:
            self.active_dof_mask = wp.ones(self.dim, dtype=wp.float32, device=device)
        elif self.active_dof_mask.shape[0] != self.dim:
            raise ValueError(f"active_dof_mask must have length {self.dim}")
        total = self.instances * self.num_particles
        indices = np.repeat(np.arange(self.instances, dtype=np.int32), self.num_particles)
        self.expand_idx = wp.array(indices, dtype=wp.int32, device=device)
        self.particle_var = mean_var.gather(self.expand_idx)
        self.noise = wp.zeros((total, self.dim), dtype=wp.float32, device=device)
        self.std = wp.full((self.instances, self.dim), config.init_std, dtype=wp.float32, device=device)
        self.particle_residual = wp.zeros((total, residual_dim), dtype=wp.float32, device=device)
        self.particle_costs = wp.zeros((total,), dtype=wp.float32, device=device)
        self.weights = wp.zeros((self.instances, self.num_particles), dtype=wp.float32, device=device)
        self.delta = wp.zeros((self.instances, self.dim), dtype=wp.float32, device=device)
        self.mean_residual = wp.zeros((self.instances, residual_dim), dtype=wp.float32, device=device)
        self.costs = wp.zeros((self.instances,), dtype=wp.float32, device=device)


class ParticleSolver(PopulationSolver):
    """Generic position-space MPPI population solver."""

    def _build_stage_updater(self, stage_idx: int, stage_var: VarValues):
        residual_dim = sum(term.residual_dim for term in self.terms[stage_idx])
        return _MppiStage(stage_var, self.config, residual_dim, self.device)

    def _accumulate_cost(self, var: VarValues, terms, residual: wp.array, cost: wp.array):
        for term in terms:
            term.precompute(var, need_gradient=False)
        offset = 0
        for term in terms:
            if isinstance(term, SparseTask):
                term.batch_size = var.batch_size
            term.compute_weighted_residual(var, out_residual=residual, row_offset=offset)
            offset += term.residual_dim
        wp.launch(aggregate_residuals_to_costs_kernel, dim=cost.shape[0], inputs=[residual, cost], device=self.device)

    def _reset_std(self, stage: _MppiStage, scale: Union[float, wp.array] = 1.0):
        config = cast(ParticleSolverConfig, self.config)
        if isinstance(scale, wp.array):
            wp.launch(
                _mppi_fill_std_kernel,
                dim=(stage.instances, stage.dim),
                inputs=[scale, config.min_std, config.init_std],
                outputs=[stage.std],
                device=self.device,
            )
        else:
            stage.std.fill_(config.min_std + (config.init_std - config.min_std) * scale)

    def _sample_noise(self, stage: _MppiStage, iteration: int):
        config = cast(ParticleSolverConfig, self.config)
        wp.launch(
            _mppi_sample_kernel,
            dim=(stage.instances * stage.num_particles, stage.dim),
            inputs=[
                config.base_seed,
                iteration,
                stage.num_particles,
                stage.dim,
                config.lock_dim,
                stage.active_dof_mask,
                stage.std,
                stage.noise,
            ],
            device=self.device,
        )

    def _update_distribution(self, stage: _MppiStage):
        config = cast(ParticleSolverConfig, self.config)
        wp.launch(
            _mppi_weights_kernel,
            dim=stage.instances,
            inputs=[stage.particle_costs, stage.num_particles, config.beta, stage.weights],
            device=self.device,
        )
        wp.launch(
            _mppi_cov_and_delta_kernel,
            dim=(stage.instances, stage.dim),
            inputs=[
                stage.weights,
                stage.noise,
                stage.num_particles,
                config.gamma,
                config.min_std,
                stage.std,
            ],
            outputs=[stage.delta],
            device=self.device,
        )

    def _solve_stage(self, stage_idx: int, warmup: bool = False) -> wp.array:
        stage: _MppiStage = self._updaters[stage_idx]
        mean_var = self._stage_vars[stage_idx]
        terms = self.terms[stage_idx]
        config = cast(ParticleSolverConfig, self.config)
        self._reset_std(stage, config.std_scale if config.std_scale is not None else 1.0)
        for iteration in range(list(config.stages)[stage_idx].iters):
            self._sample_noise(stage, iteration)
            mean_var.gather(stage.expand_idx, out=stage.particle_var)
            stage.particle_var.integrate(stage.noise, out=stage.particle_var)
            self._accumulate_cost(stage.particle_var, terms, stage.particle_residual, stage.particle_costs)
            self._update_distribution(stage)
            mean_var.integrate(stage.delta, out=mean_var)
        self._accumulate_cost(mean_var, terms, stage.mean_residual, stage.costs)
        return stage.costs
