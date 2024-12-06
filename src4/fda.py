# License: MIT
from __future__ import annotations
from typing import Any, Mapping, Tuple
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from skfda.datasets import fetch_aemet
from skfda.exploratory.depth import ModifiedBandDepth
from skfda.exploratory.visualization import Boxplot, MagnitudeShapePlot
from skfda.exploratory.visualization.fpca import FPCAPlot
from skfda.misc.metrics import l2_distance
from skfda.ml.clustering import KMeans
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.grid import FDataGrid

import os
import pandas as pd

os.chdir(os.path.dirname(__file__))

df = pd.read_excel('Book1.xlsx', 'Sheet2')

X = FDataGrid(
    data_matrix=df.iloc[:, 1:].values.T,
    sample_points=df.iloc[:, 0].values,
)

fpca = FPCA(
    n_components=4,
    _weights=np.ones_like(df.iloc[:, 0].values)
)
fpca.fit(X)

X_red = fpca.transform(X)

fig = plt.figure(figsize=(8, 4))
FPCAPlot(
    fpca.mean_,
    fpca.components_,
    factor=50,
    fig=fig,
).plot()
plt.show()
