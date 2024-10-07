from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src2.problems.base_problem import AbstractProblem, Floats, Infeasible


class AbstractManager(ABC):
    def __init__(self, problem: AbstractProblem):
        self.fig: go.Figure = go.Figure()
        self.problem: AbstractProblem = problem
        d = {'feasibility': []}
        d.update({f'prm_{k}': [] for k in range(self.problem.n_prm)})
        d.update({f'obj_{k}': [] for k in range(self.problem.n_obj)})
        self.df: pd.DataFrame = pd.DataFrame(d)
        self.setup()

    @abstractmethod
    def setup(self):
        pass

    def _get_col(self, kind):
        return [col for col in self.df.columns if kind in col]

    def get_prm_columns(self):
        self._get_col('prm')

    def get_obj_columns(self):
        self._get_col('obj')

    @abstractmethod
    def candidate_x(self) -> Floats:
        pass

    def create_base_figure(self):
        return self.problem.create_base_figure()

    def sampling(self, show=False, save_path=None):

        x = self.candidate_x()

        # update df
        try:
            y: Floats = self.problem.objective(x)
            feasible = 1.

        except Infeasible:
            y: Floats = self.problem._raw_objective(x)
            feasible = 0.

        row = pd.DataFrame([[feasible, *x, *y]], columns=self.df.columns)
        self.df = pd.concat([self.df, row], axis=0)

        # update figure
        if (save_path is not None) or show:
            self.fig = self.create_base_figure()

            self.fig.add_trace(
                go.Scatter(
                    x=self.df['obj_0'],
                    y=self.df['obj_1'],
                    mode='markers',
                    marker=dict(
                        color=self.df['feasibility'],
                        showscale=True,
                        colorscale='portland'
                    )
                )
            )

            if save_path:
                self.fig.write_image(save_path, engine="orca")

            if show:
                self.fig.show()
