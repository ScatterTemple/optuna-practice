import numpy as np
from src2.methods.admmbo.admmo_type import (
    Problem, F, C, ADMM, Opt,
    Input, Output, Inputs, Outputs, Bounds, ColVector, InputsEachCns
)
from src2.problems.base_problem import Floats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=Warning)


class BoTrace:
    samples: Inputs


def bayesopt(_f: callable, opt: F,
             _bounds, _c: list[callable], _n_prm, _n_obj) -> tuple[Input, Output, BoTrace]:
    """function [minsample,minvalue,botrace] = bayesopt(F,opt)"""
    # この中で initial_points のサンプリングをする実装になっているが、
    # それは外に出したほうがいいと思う

    # ===== 既存コードを流用するための一時的な実装 =====
    from src2.methods.botorch_manager import BayesianManager
    from src2.problems.base_problem import AbstractProblem

    # admmo の問題を既存コードのクラスに変換
    class Temp(AbstractProblem):

        @property
        def bounds(self) -> list[list[float, float]] or Floats:
            return _bounds

        def _raw_objective(self, _x: Floats) -> Floats:
            return [-v for v in _f(_x)]  # 最小化問題に変換

        def _hidden_constraint(self, _x: Floats) -> bool:
            return all([_ci(_x) <= 0 for _ci in _c])

    # problem と sampler を設定
    temp_problem = Temp(_n_prm, _n_obj)
    sampler = BayesianManager(temp_problem)

    # 提供された inputs に基づいて initial sampling
    inputs: Inputs = []
    outputs: Outputs = []
    for x in tqdm(opt.initial_points, '(一時的な実装による)bayesopt 初期サンプリングの再実行中...'):
        # assert isinstance(x, Input)

        # if temp_problem._hidden_constraint(x):
        if True:
            # feasible
            y = temp_problem._raw_objective(x)
            inputs.append(x)
            outputs.append(y)
        else:
            y = temp_problem._raw_objective(initial_points[0])
            inputs.append(initial_points[0])
            outputs.append(y)

    nd_inputs = np.array(inputs)
    nd_outputs = np.array(outputs)

    # initial sampling を sampler に反映、
    # これでベイズ最適化ができるようになった。
    for i in range(_n_prm):
        sampler.df[f'prm_{i}'] = nd_inputs[:, i]
    for i in range(_n_obj):
        sampler.df[f'obj_{i}'] = nd_outputs[:, i]
    sampler.df['feasibility'] = True

    # ベイズ最適化
    for _ in tqdm(range(opt.max_iters), 'bayesopt 最適化中...'):
        x = sampler.sampling()
        inputs.append(x)

    # 第二戻り値は捨てている
    # 第三は Samples があればよさそう。
    botrace = BoTrace()
    botrace.samples = np.array(inputs)
    return x, None, botrace


def bayesfeas(problem: Problem, opt: Opt, j: int,
              _bounds, _n_prm) -> tuple[Input, ColVector, BoTrace]:
    """function [zmin,h_min,botrace] = bayesfeas(problem,opt,const_num)"""
    # こっちは比較的短いが、論文には 普通の EI を使ってもいいと書いていたので
    # いったんそれで実装してもいいと思う。

    # j 番目の callable のみ処理を通す
    _c: callable  # if feasible: <= 0, so want to minimize this.
    for _j, _c in enumerate(problem.C):
        if j != j:
            continue
        else:
            break

    # ===== 既存コードを流用するための一時的な実装 =====
    import pandas as pd
    from src2.methods.botorch_manager import BayesianManager
    from src2.problems.base_problem import AbstractProblem

    # admmo の問題を既存コードのクラスに変換
    class Temp(AbstractProblem):

        @property
        def bounds(self) -> list[list[float, float]] or Floats:
            return _bounds

        def _raw_objective(self, _x: Floats) -> Floats:
            return [-_c(_x)]  # botorch assumes maximization problem

        def _hidden_constraint(self, _x: Floats) -> bool:
            return True  # 拘束の評価は常に可能

    # problem と sampler を設定
    temp_problem = Temp(_n_prm, 1)
    sampler = BayesianManager(temp_problem)

    # 提供された inputs に基づいて initial sampling
    inputs: Inputs = []
    outputs: Outputs = []
    for x in tqdm(opt.c[j].initial_points, '(一時的な実装による)bayesfeas サンプリングの再実行中...'):
        # assert isinstance(x, Input)

        if temp_problem._hidden_constraint(x):
            # feasible
            y = temp_problem._raw_objective(x)
            inputs.append(x)
            outputs.append(y)

    nd_inputs = np.array(inputs)
    nd_outputs = np.array(outputs)

    # initial sampling を sampler に反映、
    # これでベイズ最適化ができるようになった。
    for i in range(_n_prm):
        sampler.df[f'prm_{i}'] = nd_inputs[:, i]
    for i in range(1):
        sampler.df[f'obj_{i}'] = nd_outputs[:, i]
    sampler.df['feasibility'] = True

    # ベイズ最適化
    for _ in tqdm(range(opt.f.max_iters), 'bayesfeas 最適化中...'):
        # candidate x to maximize -_c()
        x = sampler.sampling()
        inputs.append(x)

    # 第二戻り値は捨てている
    # 第三は Samples があればよさそう。
    botrace = BoTrace()
    botrace.samples = np.array(inputs)
    return x, None, botrace


def ADMMBO(problem: Problem, opt: Opt) -> tuple[Inputs, Input, Output]:
    """
    :return: Samples, best_input, best_output
    """
    # parameter?
    ABSTOL = 0.05
    RELTOL = 0.05
    relaxp = 1.5
    relaxd = 1.5
    C_check = np.ones(len(problem.c))
    con = np.zeros(len(problem.c))
    mu = 10
    thai = 2
    Samples = []
    tolC = 0.01

    # Rebuilding the functions & evaluating the function values for an infeasible point
    _f: callable = problem.F
    _c: list[callable] = problem.C
    best_point = problem.InfPoint
    best_F = 1000 * _f(best_point) * np.sign(_f(best_point))  # FIXME: 多目的最適化に対応させる
    best_C = np.ones(len(problem.c))
    for s in range(len(problem.c)):
        best_C[s] = _c[s](best_point) * 1000 * np.sign(_c[s](best_point))

    # ADMM Main Loop
    oldSamplesNumC = [None for _ in problem.c]
    for i in range(opt.ADMM.max_iters):
        print(f"Solving the Optimality subproblem at outer iteration {i + 1} of ADMMBO:")

        # X-update
        ybar = np.mean(np.stack(opt.f.y, axis=2), axis=2)
        zbar = np.mean(np.stack(opt.f.z, axis=2), axis=2)

        def AL_X(X):
            # @(X) F(X)+(length(problem.c)*opt.ADMM.rho/2)*norm(X-(zbar)+(ybar/opt.ADMM.rho))^2;
            return _f(X) + (len(problem.c) * opt.ADMM.rho / 2) * np.linalg.norm(X - zbar + (ybar / opt.ADMM.rho)) ** 2

        if i == 0:
            opt.f.max_iters = len(opt.f.initial_points) + opt.f.step_iters
        else:
            opt.f.max_iters = len(opt.f.initial_points) + opt.f.reduced_step_iters

        # FIXME: 手動で initial_points の処理は入れているので、こうしてもよい？
        if i == 0:
            opt.f.max_iters = opt.f.step_iters
        else:
            opt.f.max_iters = opt.f.reduced_step_iters

        # Optimizing the AUGMENTED LAGRANGIAN wrt X using BO
        xmin, _, T_u = bayesopt(AL_X, opt.f,
                                problem.bounds, problem.C, opt.f.dims, opt._n_obj)
        # ここに XX(i, :) = xmin; というのがあるが、何かわからない

        # Updating Samples set
        oldSamplesNum = T_u.samples.shape[0]
        if i == 0:
            Samples.extend(T_u.samples)  # リストすべて
        else:
            Samples.extend(T_u.samples[oldSamplesNum:])  # 重複分を除く

        # updating x*_i for Z-update and Y-updates & gathering the observed data
        opt.f.x = xmin  # Updating x* for Z-update and Y-updates
        opt.f.initial_points = T_u.samples  # Updating initial X points

        iter_num = f"{i + 1}st" if i == 0 else f"{i + 1}nd"

        # Updating the incumbent
        best_point, best_F, best_C = incumbent_update(
            best_point,
            best_F,
            best_C,
            xmin,
            _f,
            _c,
            tolC,
            iter_num,
            "Optimality"
        )

        # Z-update
        print(f"Solving the Feasibility subproblem at {iter_num} outer iteration of ADMMBO:")

        # Keeping track of old Z* to check (Z*(k+1)-Z*(k))
        zold = opt.f.z  # これ if の前にあるべきでは...？

        # Checking if we have already satisfied the constraint's coonvergence criterion by C_check
        if np.all(C_check):

            zmin: InputsEachCns = [None for _ in problem.c]
            T_h: list[BoTrace or None] = [None for _ in problem.c]
            for j in range(len(problem.c)):
                # Adapting the max number of BO iterations according to ADMMBO's inner loop
                if i == 0:
                    opt.c[j].max_iters = opt.c[j].step_iters
                else:
                    opt.c[j].max_iters = opt.c[j].reduced_step_iters

                # Optimizing the feasibility subproblem j^th  wrt Z using BO
                zmin[j], _, T_h[j] = bayesfeas(problem, opt, j,
                                               problem.bounds, opt.f.dims)  # Assuming bayesfeas function exists

                subproblem = f"{j + 1}st" if j == 0 else f"{j + 1}nd"
                Samples.extend(T_h[j].samples if i == 0 else T_h[j].samples[oldSamplesNumC[j]:])

                # ZZ = opt.f.z  # 何これ？原文: ZZ{j}(i,:)=opt.f.z{j};
                # Updating the Samples set based on Z-step BO results
                oldSamplesNumC[j] = T_h[j].samples.shape[0]
                opt.f.z[j] = zmin[j].reshape(-1, 1)  # Updating z* for X-update and Y-updates
                opt.c[j].initial_points = T_h[j].samples  # Updating initial Z points

                # Updating the incumbent
                best_point, best_F, best_C = incumbent_update(
                    best_point,
                    best_F,
                    best_C,
                    zmin[j],
                    _f,
                    _c,
                    tolC,
                    iter_num,
                    subproblem
                )

                # Y-update
                # (ここ output 次元が 1 であることが前提のコードになっている気がする)
                ymin = opt.f.y[j] + opt.ADMM.rho * (xmin - zmin[j])
                opt.f.y[j] = ymin
                # clear ymin

        # Check termination condition
        history = dict(
            r_norm=[], s_norm=[], eps_pri=[], eps_dual=[]
        )
        for j in range(len(problem.c)):
            history['r_norm'].append(np.linalg.norm(opt.f.x - opt.f.z[j]))
            history['s_norm'].append(np.linalg.norm(-opt.ADMM.rho * (opt.f.z[j] - zold[j])))
            history['eps_pri'].append(np.sqrt(opt.f.dims) * ABSTOL + relaxp * RELTOL * max(np.linalg.norm(opt.f.x), np.linalg.norm(-opt.f.z[j])),)
            history['eps_dual'].append(np.sqrt(opt.f.dims) * ABSTOL + relaxd * RELTOL * np.linalg.norm(opt.ADMM.rho * opt.f.y[j]))

            if history['r_norm'][j] < history['eps_pri'][j] and history['s_norm'][j] < history['eps_dual'][j]:
                con[j] = 1

        # Checking ADMM convergence
        if np.all(con):
            print(f"It takes {i+1} ADMM iterations to converge.")
            break

        # Updating penalty parameter
        if np.mean(history['r_norm']) > mu * np.mean(history['s_norm']):
            opt.ADMM.rho *= thai
            print("rho is increased")

        elif mu * np.mean(history['r_norm']) < np.mean(history['s_norm']):
            opt.ADMM.rho /= thai
            print("rho is decreased")

        else:
            print("rho is unchanged")

    return Samples, best_point, best_F


def incumbent_update(
        best_point,
        best_F,
        best_C,
        current_point,
        _f,
        _c,
        tolC,
        iter_num,
        subproblem
):
    current_F = _f(current_point)
    current_C = np.array([_c[s](current_point) for s in range(len(_c))])

    if np.sum(current_C > tolC) and np.sum(best_C > tolC):
        print(f"No feasible point is found yet after {iter_num} ADMMBO iteration during {subproblem} subproblem!")
        return best_point, best_F, best_C
    elif np.all(current_C <= tolC) and np.all(current_F <= best_F):  # 多目的最適化に完全に対応するには、これが pareto front かどうかを考えるべきである。
        print(f"The incumbent is updated & the best feasible observed value after {iter_num} ADMMBO iteration during {subproblem} subproblem is {current_F}.")
        return current_point, current_F, current_C
    else:
        print(f"The incumbent is NOT updated at {iter_num} ADMMBO iteration during {subproblem} subproblem.")
        return best_point, best_F, best_C


if __name__ == '__main__':

    # ===== 問題設定 =====
    # # spiral
    # initial_points = np.array([
    #     np.random.rand(100),  # r
    #     np.random.rand(100) * np.pi * 2,  # theta
    # ]).T
    #
    # from src2.problems.spiral import Spiral, COEF
    # g_p = Spiral()
    #
    # def upper(x):
    #     r, theta = x
    #     return COEF * r - theta
    #
    # def lower(x):
    #     r, theta = x
    #     return theta - 2 * COEF * r

    # gravity
    from src2.problems.gravity import Gravity
    g_p = Gravity()

    initial_points = np.array([
        np.random.rand(100) * 2 - 1,  # x0
        np.random.rand(100) * 2 - 1,  # x1
    ]).T

    # ===== admoo =====
    rho = .1
    M = 50
    initial_bud_f = 10
    final_bud = 5
    initial_num = 2
    y = 0
    tolC = 0.01
    initial_bud_c = 50

    p = Problem()
    p.f = 'spiral'
    p.f_parameters = []
    p.F = lambda x: np.array([-v for v in g_p._raw_objective(x)])  # 最大化問題に変換
    p.c = [f'cns{i}' for i in range(len(g_p.constraints))]
    p.c_parameters = [[], []]
    p.C = g_p.constraints  # if feasible: return value <= 0
    p.bounds = np.array(g_p.bounds)
    p.InfPoint = initial_points[0]  # initial といってもいいのでは？？

    f = F()
    f.grid_size = -1
    f.max_iters = None
    f.step_iters = 5
    f.reduced_step_iters = 5
    f.meanfunc = []
    f.covfunc = []
    f.hyp = -1
    f.dims = 2
    f.mins = p.bounds[:, 0]
    f.maxs = p.bounds[:, 1]
    f.y = np.array([np.ones((g_p.n_obj, f.dims)) * y for _ in p.c])
    f.z = np.array([f.maxs.reshape(-1, 1) for _ in p.c])
    f.rho = .1
    f.initial_points = initial_points.copy()

    c = C()
    c.grid_size = 10**5
    c.max_iters = None
    c.step_iters = 50
    c.reduced_step_iters = 5
    c.meanfunc = []
    c.covfunc = []
    c.dims = 2
    c.optimize_ei = True
    c.initial_points = initial_points.copy()

    # c2 = C()
    # c2.grid_size = 10**5
    # c2.max_iters = None
    # c2.step_iters = 50
    # c2.reduced_step_iters = 5
    # c2.meanfunc = []
    # c2.covfunc = []
    # c2.dims = 2
    # c2.optimize_ei = True
    # c2.initial_points = initial_points

    total_budget = 100 * (len(p.c) + 1)

    admm = ADMM()
    admm.max_iters = 1 + round((total_budget - (initial_bud_c * len(p.c)) - initial_bud_f) / ((len(p.c) + 1) * final_bud))
    admm.rho = rho
    admm.M = M

    o = Opt()
    o.f = f
    o.c = [c]  # , c2
    o.ADMM = admm
    o._n_obj = g_p.n_obj

    _, candidate_x, _ = ADMMBO(p, o)

    print(candidate_x)
