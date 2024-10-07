from src2.problems.base_problem import AbstractProblem, Floats

import numpy as np
from numpy import sin, cos, pi

COEF = 5 / 12 * pi

Lower, Upper = float, float


class Spiral(AbstractProblem):

    def __init__(self):
        super().__init__(2, 2)

    @property
    def bounds(self) -> list[list[Lower | Upper]]:
        return [[0., 1.], [0., 2 * pi]]

    def _raw_objective(self, x: Floats) -> Floats:
        r, theta = x
        return r * cos(theta), r * sin(theta)

    def _hidden_constraint(self, x: Floats) -> bool:
        r, theta = x
        c1 = COEF * r - theta  # <= 0 is feasible
        c2 = theta - 2 * COEF * r  # <= 0 is feasible
        return c1 <= 0 and c2 <= 0
