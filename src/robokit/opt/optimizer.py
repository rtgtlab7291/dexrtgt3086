import abc
import logging
from dataclasses import dataclass
from typing import Optional, Sequence, TypeVar, Union

from robokit.opt.variables import Var
from robokit.terms import Term


logger = logging.getLogger("robokit")

VarT = TypeVar("VarT", bound=Var)


@dataclass(frozen=True)
class OptimizerConfig:
    # ----- Stopping criteria -----
    max_iter: int = 100
    use_early_stopping: bool = True
    early_stopping_interval: int = 1
    cost_tol: float = 1e-9
    gradient_tol: float = 1e-9
    velocity_tol: float = 1e-9

    # ----- Levenberg-Marquardt parameters -----
    lm_lambda: float = 1e-3
    lambda_factor: float = 2.0
    lambda_min: float = 1e-5
    lambda_max: float = 1e10

    # ----- Misc -----
    verbose: bool = False


class Optimizer(abc.ABC):
    terms: Sequence[Term]
    config: OptimizerConfig

    def __init__(self, terms: Union[Term, Sequence[Term]], config: Optional[OptimizerConfig] = None):
        if not isinstance(terms, Sequence):
            terms = [terms]
        self.terms = terms
        if config is None:
            config = OptimizerConfig()
        self.config = config

    @abc.abstractmethod
    def solve(self, var: VarT) -> VarT: ...
