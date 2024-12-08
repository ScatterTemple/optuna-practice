import random
import numpy
import chaospy
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from scipy.optimize import minimize, NonlinearConstraint


print('===== seed =====')
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


print('===== テスト関数 =====')
def problem(x: numpy.ndarray):
    return (x**2).sum(axis=1)


print('===== training data =====')
dim = 2
x = numpy.random.rand(100, dim) * 2 - 1
bounds = [[-1, 1]] * dim
y = problem(x)


print('===== training =====')
def tensor(x):
    if isinstance(x, list):
        x = numpy.array(x)
    return torch.tensor(x, device='cpu', dtype=torch.float64)


X = tensor(x)
Y = tensor(y).unsqueeze(1)
B = tensor(bounds).t()

model = SingleTaskGP(
    train_X=X,
    train_Y=Y,
    train_Yvar=1e-4*torch.ones_like(Y),
    input_transform=Normalize(d=X.shape[-1], bounds=B),
    outcome_transform=Standardize(m=Y.shape[-1])
)

mll = ExactMarginalLogLikelihood(model.likelihood, model)

fit_gpytorch_mll(mll)


print('===== optimize =====')
# surrogate function from surrogete model
def surrogate_func(x):
    if len(x.shape) == 1:
        X = tensor([x])
        return model.posterior(X).mean.detach().squeeze(1).numpy()

    elif len(x.shape) == 2:
        X = tensor(x)
        return model.posterior(X).mean.detach().numpy()

    else:
        raise RuntimeError


# optimize
...  # 略
opt_x = [0 for _ in range(dim)]

print(surrogate_func(numpy.array(opt_x)))


print('===== sensitivity =====')
# create normal distribution for each input
x_normals = [chaospy.Normal(mu=_x, sigma=1) for _x in opt_x]
x_distribution = chaospy.J(*x_normals)  # ?

# sampling from distribution and mapping them to the objective space
x_samples: numpy.ndarray = x_distribution.sample(numpy.array([1000]), rule="sobol")
y_samples: numpy.ndarray = surrogate_func(x_samples.T)

# noinspection PyTypeChecker
# polynomical expansion of the input distribution
x_distribution_expansion: chaospy.ndpoly = chaospy.generate_expansion(
    8,  # Order of polynomial expansion.
    x_distribution
)

# noinspection PyTypeChecker
# polynomial expansion of surrogate function
surrogate_func_expansion: chaospy.ndpoly = chaospy.fit_regression(
    x_distribution_expansion,  # polynomial expansion of the original input distribution
    x_samples,  # samples from original input distribution
    y_samples,  # mapped samples
)

# 1st order Sobol' index (VarXi / VarY)
sobol_indices: numpy.ndarray = chaospy.Sens_m(
    surrogate_func_expansion,
    x_distribution
)

# Total Sobol' index
sobol_indices_t: numpy.ndarray = chaospy.Sens_t(
    surrogate_func_expansion,
    x_distribution
)


print('===== reverse-MORDO =====')
# これが目的
y_mean = numpy.array([0])
y_std = numpy.array([1])
y_distribution = chaospy.Normal(y_mean, y_std)

# 入力変数を定義
x0_mean = [0] * dim
x0_std = [0.1] * dim
x0 = numpy.array([*x0_mean, *x0_std])
bounds0 = bounds.copy()
bounds0.extend([[0, numpy.inf] * dim])


class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self, *args, **kwargs):
        print(f'callback count: {self.count}')
        self.count += 1


# 目的関数または拘束関数
def y_distributionの違反(y_values):
    alpha = 0.001  # 許容確率
    cdf = y_distribution.cdf(y_values)  # alpha ~ 1-alpha なら良い
    lower = cdf - alpha  # > 0 is OK
    upper = (1 - alpha) - cdf  # > 0 is OK
    return lower.min(), upper.min()


def make_y_values(x_distribution_: chaospy.Distribution):
    x_samples_ = x_distribution_.sample(numpy.array([999]), rule='grid')
    return surrogate_func(x_samples_.T)


def make_x_distribution(x_):
    x_mean, x_std = x_[:dim], x_[dim:]
    return chaospy.J(*[chaospy.Normal(*ms) for ms in zip(x_mean, x_std)])


# # ===== これだとダメ。 =====
# def objective(x_):
#     x_distribution_ = make_x_distribution(x_)
#     y_values = make_y_values(x_distribution_)
#     lower, upper = y_distributionの違反(y_values)
#     return lower ** 2 + upper ** 2


def objective(x_):
    x_std = x_[dim:]
    return x_std.sum()


class Constraint:
    def __init__(self):
        self.lower = None
        self.upper = None
        self.buff_x_ = None

    def is_already_evaluated(self, x_):
        if self.buff_x_ is None:
            return False
        else:
            return numpy.allclose(x_, self.buff_x_)

    def calc(self, x_):
        self.buff_x_ = x_
        x_distribution_ = make_x_distribution(x_)
        y_values = make_y_values(x_distribution_)
        self.lower, self.upper = y_distributionの違反(y_values)

    def constraint_lower(self, x_):
        if not self.is_already_evaluated(x_):
            self.calc(x_)
        return self.lower

    def constraint_upper(self, x_):
        if not self.is_already_evaluated(x_):
            self.calc(x_)
        return self.upper


cns = Constraint()

# ===== 動いたけど超長い、正規化とかしないと面倒 =====
# # noinspection PyTypeChecker
# opt = minimize(
#     fun=objective,
#     x0=x0,
#     # # for ‘trust-constr’ and ‘cobyqa’
#     # method='trust-constr',
#     # constraints=[
#     #     NonlinearConstraint(
#     #         fun=cns.constraint_lower,
#     #         keep_feasible=False,
#     #         lb=0,
#     #         ub=None,
#     #     ),
#     #     NonlinearConstraint(
#     #         fun=cns.constraint_upper,
#     #         keep_feasible=False,
#     #         lb=0,
#     #         ub=None,
#     #     ),
#     # ],
#     # for COBYLA, SLSQP
#     method='SLSQP',
#     constraints=[
#         dict(type='ineq', fun=cns.constraint_lower),  # to be non-negative
#         dict(type='ineq', fun=cns.constraint_upper),  # to be non-negative
#     ],
#     callback=Counter()
# )
# print(f'{opt.success=}')
# print(opt.x)
# print(opt.fun)


import optuna


def o_objective(trial: optuna.Trial):
    x = []
    s = []
    for i in range(dim):
        x.append(trial.suggest_float(f'm{i}', -1, 1))
    for i in range(dim):
        s_ = trial.suggest_float(f's{i}', 0, 2)  # bounds の range
        x.append(s_)
        s.append(s_)

    lower, upper = cns.constraint_lower(x), cns.constraint_upper(x)
    trial.set_user_attr('constraint', (-lower, -upper))  # < 0 is feasible

    return tuple(s)


def o_constraint(trial: optuna.trial.FrozenTrial):
    return trial.user_attrs['constraint']


study = optuna.create_study(
    sampler=optuna.samplers.NSGAIISampler(
        constraints_func=o_constraint,
    ),
    directions=['maximize'] * dim
)

study.optimize(o_objective, timeout=120)


m = numpy.empty(shape=(0, dim))
s = numpy.empty(shape=(0, dim))
for t in study.best_trials:
    # print(t.params, t.values)
    # print(t.user_attrs['constraint'])

    m_ = numpy.empty(shape=(1, dim))
    s_ = numpy.empty(shape=(1, dim))
    for k, v in t.params.items():
        for i in range(dim):
            if k == f'm{i}':
                m_[0, i] = v
            if k == f's{i}':
                s_[0, i] = v
    m = numpy.concat([m, m_], axis=0)
    s = numpy.concat([s, s_], axis=0)

    print((m_**2).sum())
    # print((s_**2).sum())


import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(
    mode='markers',
    x=s[:, 0], y=s[:, 1],
))
fig.show()

print('breakpoint')


# # # ===== robust =====
# y_mean: numpy.ndarray = chaospy.E(surrogate_func_expansion, x_distribution)  # 極値の場合、ちょっと直感的ではない。出力分布の真ん中であって入力分布の写像にならないから。
# y_std: numpy.ndarray = chaospy.Std(surrogate_func_expansion, x_distribution)



#
#
# # ===== inv-robust =====
# # これをまず固定する
# y_mean = numpy.array([0])
# y_std = numpy.array([1])
# y_distribution = chaospy.Normal(y_mean, y_std)
#
#
# # 逆問題なので、入力した結果出てくる分布が上の分布に一致するような x を考える
# # もっと言うと、x の分散を最大化したいので上は拘束条件にして x の std の和を最大にしたい
# def constriant(y_distribution_samples) -> float:
#     # y_distribution_amples が y_distribution のどれだけ内側にあるかを返す。
#     # y_distribution_samples に対応する y_distribution の pdf の値が
#     # y_distribution の  99.9% にかかるかどうかを確認する。
#     # y_distribution は Normal（したがって単峰性）なので単純に積分値で考えてよい。
#     alpha = 0.001  # 無視する確率
#     lower = y_distribution.cdf(y_distribution_samples) - alpha  # > 0 is feasible
#     upper = (1. - alpha) - y_distribution.cdf(y_distribution_samples)  # > 0 is feasible
#     return numpy.min([lower, upper])  # > 0 is feasible
#
#
# def get_x_distribution_(x_mean, x_std) -> chaospy.Distribution:
#     # 本当は GaussianMixture など多峰性関数のほうがいいだろう
#     x_normals_ = [chaospy.Normal(mu=m, sigma=s) for m, s in zip(x_mean, x_std)]
#     x_distribution_ = chaospy.J(*x_normals_)
#     return x_distribution_
#
#
# def constraint_wrapper(x):
#     x_mean, x_std = x[:dim], x[dim:]
#     x_distribution_ = get_x_distribution_(x_mean, x_std)
#     x_samples = x_distribution_.sample(numpy.array([100]), rule='sobol')
#     y_samples = surrogate_func(x_samples.T)
#     return constriant(y_samples)  # > 0 is feasible
#
#
# def get_std(x):
#     x_mean, x_std = x[:dim], x[dim:]
#     return -x_std.sum()  # want to maximize x_std
#
#
# count = 0
#
#
# def counter(*args, **kwargs):
#     global count
#     count += 1
#     print(f'===== {count} =====')
#     print(args, kwargs)
#     print()
#
#
# bounds2 = bounds.copy()
# # bounds2.append([0, numpy.inf])
# # bounds2.append([0, numpy.inf])
# # bounds2.append([0, numpy.inf])
# bounds2.append([0, 1])
# bounds2.append([0, 1])
# bounds2.append([0, 1])
# # noinspection PyTypeChecker
# opt = minimize(
#     fun=get_std,
#     x0=numpy.array([0, 0, 0, 0.01, 0.01, 0.01]),  # parametrs of input distribution
#     bounds=bounds2,
#     # method='COBYLA',  # COBYLA, COBYQA, SLSQP and trust-constr
#     # method='COBYQA',  # COBYLA, COBYQA, SLSQP and trust-constr
#     method='SLSQP',  # COBYLA, COBYQA, SLSQP and trust-constr
#     # method='trust-constr',  # COBYLA, COBYQA, SLSQP and trust-constr
#     constraints=[
#         NonlinearConstraint(
#             constraint_wrapper,
#             lb=0,
#             # ub=numpy.inf,
#             # ub=None,
#             ub=10,
#             keep_feasible=True,
#         )
#     ],
#     callback=counter,
#     # args=(,),
# )
#
# print(opt)
# print(opt.x)
#
#
# # ほかの考え方として、違反レベルを ±0 にするための
# # 最適化を行うのでもいいかもしれない。
#
# def objective(x) -> float:
#     x_mean, x_std = x[:dim], x[dim:]
#     x_distribution_ = get_x_distribution_(x_mean, x_std)
#     x_samples = x_distribution_.sample(numpy.array([100]), rule='sobol')
#     y_distribution_samples = surrogate_func(x_samples.T)
#     alpha = 0.001  # 無視する確率
#     lower = y_distribution.cdf(y_distribution_samples) - alpha  # > 0 is feasible
#     upper = (1. - alpha) - y_distribution.cdf(y_distribution_samples)  # > 0 is feasible
#     return numpy.min(
#         [
#             lower.min() ** 2 + upper.min() ** 2,
#             lower.min() ** 2,
#             upper.min() ** 2
#         ]
#     )
#
#
# # bounds2 = bounds.copy()
# # bounds2.append([0, numpy.inf])
# # bounds2.append([0, numpy.inf])
# # bounds2.append([0, numpy.inf])
# # # bounds2.append([0, 1])
# # # bounds2.append([0, 1])
# # # bounds2.append([0, 1])
# # # noinspection PyTypeChecker
# # opt = minimize(
# #     fun=objective,
# #     x0=numpy.array([0, 0, 0, 0.01, 0.01, 0.01]),  # parametrs of input distribution
# #     bounds=bounds2,
# #     callback=counter,
# #     method='L-BFGS-B',
# #     # args=(,),
# # )
# #
# # print(opt)
# # print(opt.x)
# # print(opt.fun)
#
#
# import optuna
#
#
# def obj(trial: optuna.Trial):
#     x = []
#     for i in range(dim):
#         x.append(trial.suggest_float(f'm{i}', -1, 1))
#     for i in range(dim):
#         x.append(trial.suggest_float(f's{i}', 0, 2))
#     x = numpy.array(x)
#     cns = constraint_wrapper(x)
#     trial.set_user_attr('constraint', [-cns])  # < 0 is feasible
#     return tuple(x[dim:])
#
#
# def cons(trial):
#     return trial.user_attrs['constraint']
#
#
# study = optuna.create_study(
#     sampler=optuna.samplers.NSGAIISampler(
#         constraints_func=cons
#     ),
#     # sampler=optuna.samplers.TPESampler(
#     #     constraints_func=cons
#     # ),
#     directions=['maximize']*dim,
# )
# study.optimize(obj, timeout=30)
# parameters = []
# # print(study.best_params)
# for t in study.best_trials:
#     print(t.params)
#     x = tuple(t.params.values())
#     print(constraint_wrapper(x))
#     parameters.append(x[dim:])
#
#
# p = numpy.array(parameters)
#
# import plotly.express as px
# import plotly.graph_objects as go
#
# fig = go.Figure()
# fig.add_trace(
#     go.Scatter(
#         x=p[:, 0],
#         y=p[:, 1],
#         mode='markers',
#     )
# )
# fig.show()
#
# # うまくいかない
