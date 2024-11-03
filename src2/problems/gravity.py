from src2.problems.base_problem import AbstractProblem, Floats

import numpy as np


Lower, Upper = float, float


class Gravity(AbstractProblem):

    def __init__(self):
        super().__init__(2, 1)

    @property
    def bounds(self) -> list[list[Lower | Upper]]:
        return [[-1, 1], [-1, 1]]

    def _raw_objective(self, x: Floats) -> Floats:
        return [-(np.array(x) ** 2).sum()]  # (0, 0) で最大

    def _hidden_constraint(self, x: Floats) -> bool:
        x0, x1 = x
        return x1 > 0.5

    def objective(self, x: Floats) -> Floats:
        return self._raw_objective(x)

    @property
    def constraints(self):
        return [self.c1]

    def c1(self, x) -> float:
        x0, x1 = x
        return 0.5 - x0  # <= 0 is feasible
