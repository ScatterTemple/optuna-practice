from typing import Callable

from functools import partial

import numpy as np
from optuna.study import Study
import torch
from torch import Tensor
from botorch.acquisition import AcquisitionFunction

from optuna._transform import _SearchSpaceTransform

from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement

Constraint = Callable[[dict], float]  # <= 0 is feasible (based on optuna)


def do_patch(
        study: Study,
        constraints: list[Constraint],
        add_penalty=True,
        add_constraints=True,
        replace_qExpectedHypervolumeImprovement=True,
):
    """BoTorchSampler の optimize_acqf をパッチします。"""
    import optuna_integration
    from optuna_integration import version

    if int(version.__version__.split('.')[0]) >= 4:
        target_fun = optuna_integration.botorch.botorch.optimize_acqf
    else:
        target_fun = optuna_integration.botorch.optimize_acqf

    new_fun: callable = OptimizeReplacedACQF(target_fun)
    new_fun.set_constraints(constraints)
    new_fun.set_study(study)
    new_fun.set(add_penalty, add_constraints)

    if int(version.__version__.split('.')[0]) >= 4:
        optuna_integration.botorch.botorch.optimize_acqf = new_fun
    else:
        optuna_integration.botorch.optimize_acqf = new_fun

    if replace_qExpectedHypervolumeImprovement:
        if int(version.__version__.split('.')[0]) >= 4:
            optuna_integration.botorch.botorch.monte_carlo.qExpectedHypervolumeImprovement = qLogExpectedHypervolumeImprovement
        else:
            optuna_integration.botorch.monte_carlo.qExpectedHypervolumeImprovement = qLogExpectedHypervolumeImprovement


class GeneralFunctionWithForwardDifference(torch.autograd.Function):
    """自作関数を pytorch で自動微分するためのクラスです。

    ユーザー定義関数を botorch 形式に変換する過程で微分の計算ができなくなるのでこれが必要です。
    """

    @staticmethod
    def forward(ctx, f, xs):
        ys = f(xs)
        ctx.save_for_backward(xs, ys)
        ctx.f = f
        return ys

    @staticmethod
    def backward(ctx, grad_output):
        xs, ys = ctx.saved_tensors
        f = ctx.f
        dx = 0.001  # 入力は normalized なので決め打ちでよい
        diff = []
        xs = xs.detach()  # xs に余計な計算履歴を残さないために、detachする。
        for i in range(len(xs)):
            xs[i] += dx
            diff.append(torch.sum(grad_output * (f(xs) - ys)))
            xs[i] -= dx
        diff = torch.tensor(diff) / dx
        return None, diff


class ConvertedConstraint:
    """ユーザーが定義した Constraint を botorch で処理できる形式に変換します。

    `callable()` は形状 `d` の 1 次元テンソルを受け取り、スカラーを返します。
    """

    def __init__(self, constraint: Constraint, study: Study):
        self._constraint: Constraint = constraint
        self.study = study

    def __call__(self, x: Tensor) -> Tensor:
        """optimize_acqf() に渡される非線形拘束関数の処理です。

        Args:
            x (Tensor): Normalized parameters. Its length is d (== len(prm)).

        Returns:
            float Tensor. >= 0 is feasible.

        """
        trial = self.study.trials[-1]
        search_space = self.study.sampler.infer_relative_search_space(self.study, trial)
        trans = _SearchSpaceTransform(search_space, transform_0_1=True, transform_log=False, transform_step=False)

        return Tensor([-self._constraint(trans.untransform(x.detach().numpy()))])


def is_feasible(study: Study, cns: Constraint, values: np.ndarray) -> bool:
    """Evaluate given constraint function is feasible or not against the given X.

    Args:
        study
        cns ():
        values (np.ndarray): Normalized values of parameter set.

    """
    trial = study.trials[-1]
    search_space = study.sampler.infer_relative_search_space(study, trial)
    trans = _SearchSpaceTransform(search_space, transform_0_1=True, transform_log=False, transform_step=False)

    return cns(trans.untransform(values)) <= 0


class NonlinearInequalityConstraints:
    """botorch の optimize_acqf に parameter constraints を設定するための引数を作成します。"""

    def __init__(self, study: Study, constraints: list[Constraint]):
        self._study = study
        self._constraints = constraints
        self._nonlinear_inequality_constraints = []
        for c in self._constraints:
            item = (lambda x: GeneralFunctionWithForwardDifference.apply(ConvertedConstraint(c, self._study), x), True)
            self._nonlinear_inequality_constraints.append(item)

    def _filter_feasible_conditions(self, ic_batch):
        # List to store feasible initial conditions
        feasible_ic_list = []

        for each_num_restarts in ic_batch:
            feasible_q_list = []
            for each_q in each_num_restarts:
                ic: torch.Tensor = each_q  # normalized function
                if all(is_feasible(self._study, c, ic.numpy()) for c in self._constraints):
                    feasible_q_list.append(ic)  # Keep only feasible rows

            if feasible_q_list:  # Only add if there are feasible rows
                feasible_ic_list.append(torch.stack(feasible_q_list))

        # Stack feasible conditions back into tensor format
        if feasible_ic_list:
            return torch.stack(feasible_ic_list)
        else:
            return None  # Return None if none are feasible

    @staticmethod
    def _generate_random_initial_conditions(shape):
        # Generates random initial conditions with the same shape as ic_batch
        return torch.rand(shape)

    def _generate_feasible_initial_conditions(self, *args, **kwargs):
        # A `num_restarts x q x d` tensor of initial conditions.
        ic_batch = gen_batch_initial_conditions(*args, **kwargs)
        feasible_ic_batch = self._filter_feasible_conditions(ic_batch)

        while feasible_ic_batch is None:
            # Generate new random ic_batch with the same shape
            print('警告：gen_batch_initial_conditions() は feasible な初期値を提案しませんでした。'
                  'パラメータ提案を探索するための初期値をランダムに選定します。')
            random_ic_batch = self._generate_random_initial_conditions(ic_batch.shape)
            feasible_ic_batch = self._filter_feasible_conditions(random_ic_batch)

        return feasible_ic_batch

    def create_kwargs(self) -> dict:
        """
        nonlinear_inequality_constraints:
            非線形不等式制約を表すタプルのリスト。
            タプルの最初の要素は、`callable(x) >= 0` という形式の制約を表す呼び出し可能オブジェクトです。
            2 番目の要素はブール値で、点内制約の場合は `True`
            制約は後で scipy ソルバーに渡されます。
            この場合、`batch_initial_conditions` を渡す必要があります。
            非線形不等式制約を使用するには、`batch_limit` を 1 に設定する必要もあります。
                これは、`options` で指定されていない場合は自動的に行われます。
        """
        return dict(
            q=1,
            options=dict(
                batch_limit=1,
            ),
            nonlinear_inequality_constraints=self._nonlinear_inequality_constraints,
            ic_generator=self._generate_feasible_initial_conditions,
        )


class AcquisitionFunctionWithPenalty(AcquisitionFunction):
    """獲得関数に infeasible 項を追加します。"""

    # noinspection PyAttributeOutsideInit
    def set_acqf(self, acqf):
        self._acqf = acqf

    # noinspection PyAttributeOutsideInit
    def set_constraints(self, constraints):
        self._constraints: callable = constraints

    # noinspection PyAttributeOutsideInit
    def set_study(self, study: Study):
        self._study: Study = study

    def forward(self, X: "Tensor") -> "Tensor":
        """

        Args:
            X (Tensor): batch_size x 1 x n_params tensor.

        Returns:
            Tensor: batch_size tensor.

        """
        base = self._acqf.forward(X)

        row: np.ndarray
        for i, row in enumerate(X.detach().numpy()):
            if any([not is_feasible(self._study, cns, row.flatten()) for cns in self._constraints]):
                base[i] = 0.

        return base


class OptimizeReplacedACQF(partial):
    """optimize_acqf をこの partial 関数に置き換えます。"""

    # noinspection PyAttributeOutsideInit
    def set_constraints(self, constraints):
        self._constraints = constraints

    # noinspection PyAttributeOutsideInit
    def set_study(self, study: Study):
        self._study: Study = study

    # noinspection PyAttributeOutsideInit
    def set(self, add_penalty, add_constraints):
        self._add_penalty = add_penalty
        self._add_constraints = add_constraints

    def __call__(self, *args, **kwargs):
        """置き換え先の関数の処理内容です。

        kwargs を横入りして追記することで拘束を実現します。
        """

        # 獲得関数に infeasible な場合のペナルティ項を追加します。
        if self._add_penalty:
            acqf = kwargs['acq_function']
            new_acqf = AcquisitionFunctionWithPenalty(...)
            new_acqf.set_acqf(acqf)
            new_acqf.set_constraints(self._constraints)
            new_acqf.set_study(self._study)
            kwargs['acq_function'] = new_acqf

        # optimize_acqf の探索に parameter constraints を追加します。
        if self._add_constraints:
            nlic = NonlinearInequalityConstraints(self._study, self._constraints)
            kwargs.update(nlic.create_kwargs())

        # replace other arguments

        return super().__call__(*args, **kwargs)
