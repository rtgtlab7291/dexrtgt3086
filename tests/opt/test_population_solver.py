"""Structural guards for the PopulationSolver base extracted from MultiSeedSolver."""

import pytest

from robokit.opt.multi_seed_solver import MultiSeedSolver, MultiSeedSolverConfig
from robokit.opt.population_solver import PopulationSolver, PopulationSolverConfig, StageConfig


class TestPopulationSolver:
    def test_multiseed_is_population_solver(self):
        assert issubclass(MultiSeedSolver, PopulationSolver)
        assert issubclass(MultiSeedSolverConfig, PopulationSolverConfig)

    def test_base_is_abstract(self):
        # The solve seam (_build_stage_updater / _solve_stage) is abstract, so the base
        # cannot be instantiated on its own - every solver must supply an update rule.
        config = PopulationSolverConfig(stages=[StageConfig(num_seeds=1, iters=1)])
        with pytest.raises(TypeError):
            PopulationSolver(terms=[[]], config=config, device="cpu")
