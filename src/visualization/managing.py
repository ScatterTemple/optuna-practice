from abc import ABC, abstractmethod

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class OptimizationManager(ABC):
    def __init__(self):
        self.df: pd.DataFrame = pd.DataFrame()
        self.fig: go.Figure = go.Figure()

    @abstractmethod
    def sampling(self):
        pass

    @abstractmethod
    def create_figure(self):
        pass

    def run(self):
        self.sampling()
        self.create_figure()
        self.fig.show()
