from typing import Optional, Any
import os
from time import time
import optuna
from optuna.samplers import BaseSampler, TPESampler
from optuna_integration import BoTorchSampler
import plotly.graph_objects as go
from src.template.spiral import (
    objectives,
    constraints,
    directions,
    create_space_fig,
    __constraint_1,
    __constraint_2,
)
from src.botorch_sampler_patches import do_patch


def save_figure(fig, path):
    fig.write_image(path, engine="orca")


def main(
        Sampler: BaseSampler or Any = TPESampler,
        timeout: Optional[int] = None,
        n_trials: Optional[int] = 30,
        with_constraints: bool = True,
        n_startup_trials: int = 10,
        add_constraints: bool = True,
        add_penalty: bool = True,
        replace_qExpectedHypervolumeImprovement: bool = False,
        storage_name: str = None,
):

    kwargs: dict = dict(seed=42, n_startup_trials=n_startup_trials)
    if with_constraints:
        kwargs.update(dict(constraints_func=constraints))

    # noinspection PyCallingNonCallable
    sampler = Sampler(**kwargs)

    study = optuna.create_study(
        storage=f'sqlite:///{storage_name}.db' if storage_name is not None else None,
        directions=directions,
        sampler=sampler,
        load_if_exists=True,
    )

    do_patch(
        study=study,
        constraints=[__constraint_1, __constraint_2],
        add_constraints=add_constraints,
        add_penalty=add_penalty,
        replace_qExpectedHypervolumeImprovement=replace_qExpectedHypervolumeImprovement,
    )

    start = time()
    study.optimize(
        objectives,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )
    duration = time() - start

    df = study.trials_dataframe()

    fig = create_space_fig()

    fig.add_trace(
        go.Scatter(
            x=df.values_0,
            y=df.values_1,
            mode='markers',
            marker=dict(
                color=df.number[::-1],
                colorscale="gray"
            ),
        )
    )

    sampler_name = sampler.__str__()
    if issubclass(Sampler, BoTorchSampler):
        if add_penalty:
            sampler_name = 'Penalized' + sampler_name
        if add_constraints:
            sampler_name = 'Constrained' + sampler_name

    title = (f'{len(df)} samples '
             f'by {sampler_name}<BR>'
             f'{"with" if with_constraints else "without"} constraints '
             f'in {int(duration)} sec'
             )
    fig.update_layout(
        dict(
            title_text=title,
            autosize=False,
            width=500,
            height=500
        )
    )

    return fig, title


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    fig, title = main(
        TPESampler,
        n_trials=None,
        timeout=3,
        with_constraints=True
    )

    path = os.path.join('images', title.replace('<BR>', ' ') + '.svg')
    save_figure(fig, path)
