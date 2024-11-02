# ===== import =====

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any, List, Optional
import warnings

import numpy
import numpy as np
from botorch.models.model import Model
from optuna import logging
from optuna._experimental import experimental_class
from optuna._experimental import experimental_func
from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import RandomSampler
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _process_constraints_after_trial
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from packaging import version


from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective import monte_carlo
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import (
    FeasibilityWeightedMCMultiOutputObjective,
)
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.objective import ConstrainedMCObjective, PosteriorTransform
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import ModelListGP
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.sampling.list_sampler import ListSampler
import botorch.version
from torch import Tensor

if version.parse(botorch.version.version) < version.parse("0.8.0"):
    from botorch.fit import fit_gpytorch_model as fit_gpytorch_mll

    def _get_sobol_qmc_normal_sampler(num_samples: int) -> SobolQMCNormalSampler:
        return SobolQMCNormalSampler(num_samples)

else:
    from botorch.fit import fit_gpytorch_mll

    def _get_sobol_qmc_normal_sampler(num_samples: int) -> SobolQMCNormalSampler:
        return SobolQMCNormalSampler(torch.Size((num_samples,)))

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
import torch

from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import manual_seed
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import normalize
from botorch.utils.transforms import unnormalize

from botorch.acquisition.analytic import LogConstrainedExpectedImprovement
from botorch.acquisition.analytic import LogExpectedImprovement

from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    qHypervolumeKnowledgeGradient,
)

# ===== 手法の説明 =====
print('目的関数のサロゲートモデルとは別に\n'
      '破綻する / しないを判定するモデルを作成し\n'
      'それを獲得関数に反映します。\n'
      '参考：https://gpflowopt.readthedocs.io/en/latest/notebooks/constrained_bo.html\n')


# ===== 問題の設定 =====
from src.problem.spiral import COEF, obj_1, obj_2, ObjectiveError
from numpy import pi


# feasibility 評価用の関数を作成
def feasibility(r, theta):
    try:
        obj_1(r, theta)
        obj_2(r, theta)
        return 1.  # -1. or 1.
    except ObjectiveError:
        return 0.  # 1. or 0.


# ===== 学習用データ =====

# bounds
bounds = dict(r=(0, 1), theta=(0, 2*pi))

# input
prm_train =[
    dict(r=0.5, theta=COEF * 0.5 * 1.5),
    dict(r=0.25, theta=COEF * 0.25 * 1.25),
    dict(r=0.25, theta=COEF * 0.25 * 1.75),
    dict(r=0.75, theta=COEF * 0.75 * 1.25),
    dict(r=0.75, theta=COEF * 0.75 * 1.75),
]

# output
results = [(obj_1(**kw), obj_2(**kw)) for kw in prm_train]  # maximize

# input of feasibility
prm_train_f = prm_train.copy()

# output of feasibility
results_f = [(feasibility(**kw),) for kw in prm_train]  # maximize

t_bounds: torch.Tensor = torch.Tensor(tuple(bounds.values()), device='cpu').double().t()


# ===== 問題オブジェクトの設定 =====
from visualization.managing import OptimizationManager


class PoFConstraint(OptimizationManager):

    def __init__(self):
        super().__init__()

    def sampling(self):
        global prm_train, results, prm_train_f, results_f

        # ===== 学習 =====

        # feasibility
        t_x_f: torch.Tensor = torch.Tensor([tuple(kw.values()) for kw in prm_train_f], device='cpu').double()
        t_norm_x_f: torch.Tensor = normalize(t_x_f, t_bounds)
        t_y_f: torch.Tensor = torch.Tensor(results_f, device='cpu').double()

        model_f = SingleTaskGP(
            t_norm_x_f,
            t_y_f,
            train_Yvar=1e-5*torch.ones_like(t_y_f),
            outcome_transform=Standardize(m=t_y_f.size(-1)),
        )
        mll_f = ExactMarginalLogLikelihood(model_f.likelihood, model_f)
        fit_gpytorch_mll(mll_f)

        # 本番
        n_objectives = len(results[0])

        t_x: torch.Tensor = torch.Tensor([tuple(kw.values()) for kw in prm_train], device='cpu').double()
        t_norm_x: torch.Tensor = normalize(t_x, t_bounds)
        t_y: torch.Tensor = torch.Tensor(results, device='cpu').double()

        model = SingleTaskGP(
            t_norm_x,
            t_y,
            train_Yvar=1e-5*torch.ones_like(t_y),
            outcome_transform=Standardize(m=t_y.size(-1)),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # ===== 獲得関数の計算 =====

        # Approximate box decomposition similar to Ax when the number of objectives is large.
        # https://github.com/pytorch/botorch/blob/36d09a4297c2a0ff385077b7fcdd5a9d308e40cc/botorch/acquisition/multi_objective/utils.py#L46-L63
        if n_objectives > 4:
            alpha = 10 ** (-8 + n_objectives)
        else:
            alpha = 0.0

        ref_point = t_y.min(dim=0).values - 1e-8

        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=t_y, alpha=alpha)

        ref_point_list = ref_point.tolist()

        class PenalizedExpectedHypervolumeImprovementBase(ExpectedHypervolumeImprovement):
            def __init__(self, model: Model, ref_point: List[float], partitioning: NondominatedPartitioning,
                         posterior_transform: Optional[PosteriorTransform] = None, model_f=None) -> None:
                """コンストラクタに model_f を追加しただけのもの"""
                super().__init__(model, ref_point, partitioning, posterior_transform)
                self._feasibility_model: SingleTaskGP = model_f

        class SigmoidPenalizedExpectedHypervolumeImprovement(PenalizedExpectedHypervolumeImprovementBase):
            """シグモイド関数でモデルの平均値を 0 から 1 に規格化する。

            分散を考慮しない。
            これを実行するときは拘束関数の出力を 0(infeasible), 1(feasible) にすること。

            """
            # tau = 0.1
            # offset = 0.5  # 0.5  # hard sigmoid にしてみては？（0|0, 1|1 は譲らないほうがいいため）
            tau = 0.05
            offset = 0.9

            def forward(self, X: Tensor) -> Tensor:
                F = self._feasibility_model.posterior(X).mean
                sigmoided_F = 1. / (1 + torch.exp(-(F-self.offset) / self.tau))
                # [feasibility(*x[0]) for x in X]
                return super().forward(X) * sigmoided_F.prod(dim=-1).sum(dim=-1)  # 勾配も変になるかも

        class PoFPenalizedExpectedHypervolumeImprovement(PenalizedExpectedHypervolumeImprovementBase):
            """これを使う場合は拘束関数を -1(feasible) ~ 1(feasible) にすること"""

            threshold = 0  # 目安 -1 ~ 1 で、小さいほど拘束回避を強くする？

            def forward(self, X: Tensor) -> Tensor:
                # X に対して予測の平均と分散を取得
                pred = self._feasibility_model.posterior(X)
                mu, sigma = pred.mean, pred.variance

                # 平均と分散から正規分布を計算
                sigma = sigma.clamp_min(1e-4)

                # 正規分布の -∞ から 0 までを積分して pof とする
                # from scipy.stats import norm
                # pof = norm.cdf(0, loc=mu.detach().numpy(), scale=sigma.sqrt().detach().numpy())
                from torch.distributions import Normal
                normal = Normal(mu, sigma)
                pof = normal.cdf(torch.zeros_like(mu) + self.threshold)

                # 本来の獲得関数を計算
                acq = super().forward(X)

                # pof の変形
                pof = pof.flatten()

                return acq * pof

        acqf = SigmoidPenalizedExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point_list,
            partitioning=partitioning,
            model_f=model_f,
        )

        standard_bounds = torch.zeros_like(t_bounds)
        standard_bounds[1] = 1

        t_candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=1,
            num_restarts=20,
            raw_samples=1024,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
            return_best_only=False,
        )

        candidates = unnormalize(t_candidates.detach(), bounds=t_bounds).numpy().flatten()

        # ===== 拘束と目的関数の評価 =====

        # input
        prm = dict(r=candidates[0], theta=candidates[1])
        print(prm)

        # feasibility check
        new_result_f = (feasibility(**prm),)

        # append
        prm_train_f.append(prm)
        results_f.append(new_result_f)

        # if feasible, add objective
        # if new_result_f[0] == -1:
        if new_result_f[0] == 1:
            print('feasible!')

            # output
            new_result = (obj_1(**prm), obj_2(**prm))  # maximize

            # append data
            prm_train.append(prm)
            results.append(new_result)

        return model, model_f


# ===== 最適化の実施 =====

# 最適化実行
n = 15
for i in range(n):
    model, model_f = sampling()

# # 現在の履歴の表示
# for item in prm_train: print(item)
# for item in prm_train_f: print(item)
# for item in results: print(item)
# for item in results_f: print(item)

# 結果表示
import plotly.graph_objects as go
import numpy as np

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
xx, yy = np.meshgrid(x, y)
rr = np.sqrt(xx**2 + yy**2)
tt = np.arctan2(yy, xx)

X = torch.Tensor(np.stack([rr, tt], axis=2), device='cpu').double()
X_norm = normalize(X, t_bounds)
ff = model_f.posterior(X_norm).mean.detach().numpy().squeeze().astype(object)
ff[np.where(rr > 1)] = None

fig = go.Figure()

fig.add_trace(
    go.Contour(
        connectgaps=False,
        z=ff,
        x=x,
        y=y,
    )
)

r, theta = np.array([list(prm.values()) for prm in prm_train_f]).T
fig.add_trace(
    go.Scatter(
        x=r * np.cos(theta),
        y=r * np.sin(theta),
        mode='markers',
        marker=dict(color='black')
    )
)
r, theta = np.array([list(prm.values()) for prm in prm_train]).T
fig.add_trace(
    go.Scatter(
        x=r * np.cos(theta),
        y=r * np.sin(theta),
        mode='markers',
        marker=dict(color='green')
    )
)
fig.show()

for item in prm_train: print(item)
for item in prm_train_f: print(item)
for item in results: print(item)
for item in results_f: print(item)
