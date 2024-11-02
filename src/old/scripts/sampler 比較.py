import os
import datetime
from optuna.samplers import TPESampler
from optuna_integration import BoTorchSampler
from src.template.template import main, save_figure


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    # # TPE
    # fig, title = main(
    #     TPESampler,
    #     n_trials=None,
    #     timeout=5,
    #     with_constraints=True,
    #     n_startup_trials=1,
    #     add_penalty=False,
    #     add_constraints=False,
    #     # storage_name=f'',
    # )
    # path = os.path.join('images', title.replace('<BR>', ' ') + '.svg')
    # save_figure(fig, path)

    # # raw BoTorch
    # fig, title = main(
    #     BoTorchSampler,
    #     n_trials=None,
    #     timeout=60,
    #     with_constraints=True,
    #     n_startup_trials=1,
    #     add_penalty=False,
    #     add_constraints=False,
    #     use_deterministic=False,
    #     replace_qExpectedHypervolumeImprovement=False,
    #     storage_name=f'botorch_60sec',
    # )
    # path = os.path.join('images', title.replace('<BR>', ' ') + '.svg')
    # save_figure(fig, path)

    # penalized BoTorch
    fig, title = main(
        BoTorchSampler,
        n_trials=None,
        timeout=3600,
        with_constraints=True,
        n_startup_trials=1,
        add_penalty=True,
        add_constraints=False,
        use_deterministic=False,
        replace_qExpectedHypervolumeImprovement=False,
        storage_name=f'penalized_botorch_3600sec',
    )
    path = os.path.join('images', title.replace('<BR>', ' ') + '.svg')
    save_figure(fig, path)

    # penalized deterministic BoTorch
    fig, title = main(
        BoTorchSampler,
        n_trials=None,
        timeout=3600,
        with_constraints=True,
        n_startup_trials=1,
        add_penalty=True,
        add_constraints=False,
        use_deterministic=True,
        replace_qExpectedHypervolumeImprovement=False,
        storage_name=f'penalized_deterministic_botorch_3600sec',
    )
    path = os.path.join('images', title.replace('<BR>', ' ') + '.svg')
    save_figure(fig, path)

