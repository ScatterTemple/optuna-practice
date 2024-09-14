import optuna
from optuna.samplers import GPSampler
from src.template.spiral import objectives, constraints, directions


sampler = GPSampler(
    seed=42,
    deterministic_objective=True,
)

study = optuna.create_study(
    directions=directions,
    sampler=sampler,
)

study.optimize(
    objectives,
    n_trials=30,
    show_progress_bar=True,
)

# ValueError: If the study is being used for multi-objective optimization, GPSampler cannot be used.
