import numpy as np
from numpy import sin, cos, pi
from optuna.trial import Trial, FrozenTrial
import plotly.express as px
import plotly.graph_objects as go


CONSTRAINTS_KEY = 'constraints'

directions = ['maximize', 'maximize']


def objectives(trial: Trial):
    r = trial.suggest_float('r', 0, 1)
    theta = trial.suggest_float('theta', 0, 2*pi)

    c1 = _constraint_1(r, theta)
    c2 = _constraint_2(r, theta)
    trial.set_user_attr(CONSTRAINTS_KEY, (c1, c2))

    x = _objective_1(r, theta)
    y = _objective_2(r, theta)
    return x, y


def constraints(trial: FrozenTrial):
    return trial.user_attrs[CONSTRAINTS_KEY]


def _constraint_1(r, theta):
    return 5/12*pi*r - theta  # <= 0 is feasible


def __constraint_1(p: dict):
    return _constraint_1(p['r'], p['theta'])


def _constraint_2(r, theta):
    return theta - 10/12*pi*r  # <= 0 is feasible


def __constraint_2(p: dict):
    return _constraint_2(p['r'], p['theta'])


def _objective_1(r, theta):
    return r * cos(theta)


def _objective_2(r, theta):
    return r * sin(theta)


def create_space_fig():
    r = np.linspace(0, 1, 24)
    theta = np.linspace(0, 2*pi, 120)

    rr, tt = np.meshgrid(r, theta)

    x = _objective_1(rr, tt)
    y = _objective_2(rr, tt)

    c1 = _constraint_1(rr, tt)
    c2 = _constraint_2(rr, tt)

    conditions = (c1 < 0) * (c2 < 0)
    idx_feasible = np.where(conditions)
    idx_infeasible = np.where(~conditions)
    x_feasible = x[idx_feasible]
    y_feasible = y[idx_feasible]
    x_infeasible = x[idx_infeasible]
    y_infeasible = y[idx_infeasible]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_infeasible,
            y=y_infeasible,
            mode='markers',
            marker=dict(
                color='white',
                size=20,
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_feasible,
            y=y_feasible,
            mode='markers',
            marker=dict(
                color='pink',
                size=20,
            ),
        )
    )


    return fig


if __name__ == '__main__':
    create_space_fig().show()
