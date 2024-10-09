from src2.methods.base_manager import AbstractManager
from src2.problems.base_problem import Floats

import numpy as np


class RandomManager(AbstractManager):

    def setup(self):
        pass

    def candidate_x(self) -> Floats:
        out = []
        for lb, ub in self.problem.bounds:
            out.append(np.random.rand() * (ub - lb) + lb)
        return out
