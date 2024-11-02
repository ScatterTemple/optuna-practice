import numpy as np
from numpy import sin, cos, pi
import plotly.express as px
import plotly.graph_objects as go

COEF = 5/12*pi


class ObjectiveError(Exception):
    pass


def is_feasible(r, theta):
    return cns_1(r, theta) <= 0 and cns_2(r, theta) <= 0


def cns_1(r, theta):
    return COEF*r - theta  # <= 0 is feasible


def cns_2(r, theta):
    return theta - 2*COEF*r  # <= 0 is feasible


def obj_1(r, theta):
    if is_feasible(r, theta):
        return r * cos(theta)
    else:
        raise ObjectiveError


def obj_2(r, theta):
    if is_feasible(r, theta):
        return r * sin(theta)
    else:
        raise ObjectiveError


def r(x, y):
    return np.sqrt(x**2 + y**2)


def theta(x, y):
    return np.arctan2(y, x)


def polar(x, y):
    return r(x, y), theta(x, y)


def create_base_fig(return_trace=False) -> go.Figure:
    fig = go.Figure()

    n_mid_points = 40
    r_1 = np.linspace(0, 1, n_mid_points)
    theta_1 = np.linspace(0, COEF, n_mid_points)
    r_2 = np.ones(n_mid_points)
    theta_2 = np.linspace(COEF, 2*COEF, n_mid_points)
    r_3 = np.linspace(1, 0, n_mid_points)
    theta_3 = np.linspace(2*COEF, 0, n_mid_points)

    r = np.hstack((r_1, r_2, r_3))
    theta = 360 / (2*pi) * np.hstack((theta_1, theta_2, theta_3))

    trace = go.Scatterpolar(
            r=r,
            theta=theta,
            name='feasible region',
            mode='lines',
        )

    if return_trace:
        return trace

    fig.add_trace(
        trace
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1.05]),
        )
    )

    return fig


from typing import Sequence
def plot(r: Sequence[float], theta: Sequence[float]):
    fig = create_base_fig()
    fig.add_trace(
        go.Scatterpolar(
            r=r,
            theta=np.array(theta)*360/(2*pi),
            mode='markers',
            name='solutions',
        )
    )
    fig.show()


def plot_df(df):

    df['index'] = np.arange(len(df))
    df['theta'] = df['theta'] * 360 / (2*pi)

    fig = px.scatter_polar(
        df,
        r='r',
        theta='theta',
        color='index',
        start_angle=0,
        direction='counterclockwise',
        color_continuous_scale='blues',
    )
    fig.update_layout(
        coloraxis_colorbar_x=-0.15,
        polar=dict(
            radialaxis=dict(range=[0, 1.05]),
        )
    )

    base = create_base_fig(return_trace=True)
    fig.add_trace(base)

    fig.show()


if __name__ == '__main__':
    x, y = 0.1, -0.5
    r, theta = polar(x, y)
    plot([r], [theta])
