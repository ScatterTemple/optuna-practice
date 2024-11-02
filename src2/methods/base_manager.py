from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src2.problems.base_problem import AbstractProblem, Floats, Infeasible


class AbstractManager(ABC):
    def __init__(self, problem: AbstractProblem):
        self.fig: go.Figure = go.Figure()
        self.problem: AbstractProblem = problem
        d = {'feasibility': []}
        d.update({'method': []})
        d.update({f'prm_{k}': [] for k in range(self.problem.n_prm)})
        d.update({f'obj_{k}': [] for k in range(self.problem.n_obj)})
        self.df: pd.DataFrame = pd.DataFrame(d)
        self.setup()

    @abstractmethod
    def setup(self):
        pass

    def _get_col(self, kind) -> list[str]:
        return [col for col in self.df.columns if kind in col]

    def get_prm_columns(self):
        return self._get_col('prm')

    def get_obj_columns(self):
        return self._get_col('obj')

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
            feasible = True

        except Infeasible:
            y: Floats = self.problem._raw_objective(x)
            feasible = False

        method = type(self).__name__

        row = pd.DataFrame(
            [[
                feasible,
                method,
                *x,
                *y
            ]],
            columns=self.df.columns
        )
        self.df = pd.concat([self.df, row], axis=0)

        # update figure
        if (save_path is not None) or show:
            self.fig = self.create_base_figure()

            # self.fig.add_trace(
            #     go.Scatter(
            #         x=self.df['obj_0'],
            #         y=self.df['obj_1'],
            #         mode='markers',
            #         marker=dict(
            #             color=self.df['feasibility'],
            #             showscale=True,
            #             colorscale='portland'
            #         )
            #     )
            # )
            scat = px.scatter(
                self.df,
                x='obj_0',
                y='obj_1',
                symbol='feasibility',
                symbol_map={
                    True: 'circle',
                    False: 'circle-open'
                },
                color='method',
            )
            for trace in scat.data:
                self.fig.add_trace(trace)

            if save_path:
                self.fig.write_image(save_path, engine="orca")

            if show:
                self.fig.show()

        return x
