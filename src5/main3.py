import random
import numpy
import chaospy
import torch
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from scipy.optimize import minimize, NonlinearConstraint
import plotly.express as px
import plotly.graph_objects as go


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
dim = 4
bound = [-1, 1]
bounds = [bound] * dim
x = numpy.random.rand(100, dim) * (bound[1] - bound[0]) + bound[0]
y = problem(x)

print('----- x -----')
print(x[:5])

print('----- y -----')
print(y[:5].reshape(-1, 1))


print('===== create surrogate model =====')
model = GaussianProcessRegressor(
    kernel=kernels.Matern(nu=3/2)
).fit(x, y)


# surrogate function from surrogete model
def surrogate_func(x):
    return model.predict(x)


print('===== sensitivity of optimize result =====')
print('----- create the distribution around the specified x -----')
print('この手法は「グローバル感度解析」と呼ばれるが、文脈が違って、'
      'この使い方では定義域全体のうち注目する解の周辺正規分布しか'
      '見ないので、ローカルな、といっても差し支えないと思う。'
      '実際、COMSOL はそう言っている。MATLAB はそう言ってない。'
      'まず最初に、定義域全体のガウス過程回帰モデルをさらに模倣する'
      '注目する解周辺限定で精度が出る（はず）の多項式近似モデルを作る。'
      '全体のサンプルの局在は絶対ちゃんとして分布にならないので'
      '全体サロゲートモデルを多項式近似はできない（精度が出る理由がない）'
      'のでやらない。')
mu_arr = numpy.array([0.5]*dim)
sigma_arr = numpy.array([0.15]*dim)
# create normal distribution for each input
x_normals = [chaospy.Normal(mu=mu, sigma=sigma) for mu, sigma in zip(mu_arr, sigma_arr)]
x_distribution = chaospy.J(*x_normals)  # ?

# sampling from distribution and mapping them to the objective space
# to create the `global-locally-normal surrogate-surrogate-model`.


# ===== rule の挙動の調査 =====
"""
``additive_recursion``  Modulus of golden ratio samples.
``chebyshev``           Roots of first order Chebyshev polynomials.
``grid``                Regular spaced grid.
``halton``              Halton low-discrepancy sequence.
``hammersley``          Hammersley low-discrepancy sequence.
``korobov``             Korobov lattice.
``latin_hypercube``     Latin hypercube samples.
``nested_chebyshev``    Chebyshev nodes adjusted to ensure nested.
``nested_grid``         Nested regular spaced grid.
``random``              Classical (Pseudo-)Random samples.
``sobol``               Sobol low-discrepancy sequence.
"""

N = 1000


def show(rule):
    x_distribution_samples: numpy.ndarray = x_distribution.sample(numpy.array([N]), rule=rule)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_distribution_samples[0], y=x_distribution_samples[1], mode='markers'
    ))
    fig.update_layout(title=rule)
    fig.show()

# print(rule := 'random', x_distribution.sample(numpy.array([N]), rule=rule).shape); show(rule)  # (dim, N)
# print(rule := 'sobol', x_distribution.sample(numpy.array([N]), rule=rule).shape); show(rule)  # (dim, N)
# print(rule := 'additive_recursion', x_distribution.sample(numpy.array([N]), rule=rule).shape); show(rule)  # (dim, N)
# print(rule := 'chebyshev', x_distribution.sample(numpy.array([N]), rule=rule).shape); show(rule)  # (dim, N**dim)
# print(rule := 'grid', x_distribution.sample(numpy.array([N]), rule=rule).shape); show(rule)  # (dim, N**dim)
# print(rule := 'halton', x_distribution.sample(numpy.array([N]), rule=rule).shape); show(rule)  # (dim, N)
# print(rule := 'hammersley', x_distribution.sample(numpy.array([N]), rule=rule).shape); show(rule)  # (dim, N)
# print(rule := 'korobov', x_distribution.sample(numpy.array([N]), rule=rule).shape); show(rule)  # (dim, N)
# print(rule := 'latin_hypercube', x_distribution.sample(numpy.array([N]), rule=rule).shape); show(rule)  # (dim, N)
# print(rule := 'nested_chebyshev', x_distribution.sample(numpy.array([N]), rule=rule).shape); show(rule)  # (dim, 1046529)
# print(rule := 'nested_grid', x_distribution.sample(numpy.array([N]), rule=rule).shape); show(rule)  # (dim, 1046529)

# ===== rule の挙動の調査 おわり =====


x_distribution_samples: numpy.ndarray = x_distribution.sample(numpy.array([20]), rule="sobol")
y_distribution_samples: numpy.ndarray = surrogate_func(x_distribution_samples.T)

# noinspection PyTypeChecker
# basis polynomical function expansion of the input distribution
x_distribution_basis_function_expansion: chaospy.ndpoly = chaospy.generate_expansion(
    4,  # Order of polynomial expansion. dim が大きく order も大きいと分散、特に感度の計算にスタックしたかと見紛うほどの時間がかかる
    x_distribution
)

# noinspection PyTypeChecker
# create surrogate-surrogate model
local_surrogate_expansion: chaospy.ndpoly = chaospy.fit_regression(
    x_distribution_basis_function_expansion,  # *orthogonal* basis function expansion of *dist*
    x_distribution_samples,  # samples derived from *dist*
    y_distribution_samples,  # mapped samples
)

# 多項式近似モデルに対し実際の入力パラメータの分布を入れて
# 出力の期待値を計算できる（このサンプルでは入力 x は固定なので
# 出力 y のパラメータ依存性がわかる）
expected: numpy.ndarray = chaospy.E(local_surrogate_expansion, x_distribution)
theoritical_e = (sigma_arr ** 2 + mu_arr ** 2).sum()
surrogate_func([mu_arr])  # これは全然違う値になるが、分布が歪むので正しい。
print('----- 期待値 -----')
print(f'local-surrogate による予測: {float(expected.round(4))}')
print(f'非心カイ二乗分布の理論値: {float(numpy.array(theoritical_e).round(4))}')

# 同様に stddev も計算できる
var: numpy.ndarray = chaospy.Var(local_surrogate_expansion, x_distribution)
theoritical_v = (2 * sigma_arr**2 * (sigma_arr**2 + 2*mu_arr**2)).sum()
print('----- 分散 -----')
print(f'local-surrogate による予測: {float(var.round(4))}')
print(f'非心カイ二乗分布の理論値: {float(numpy.array(theoritical_v).round(4))}')

# 1st order Sobol' index (VarXi / VarY)
sobol_indices_1: numpy.ndarray = chaospy.Sens_m(
    local_surrogate_expansion,
    x_distribution
)
sobol_indices_t: numpy.ndarray = chaospy.Sens_t(
    local_surrogate_expansion,
    x_distribution
)
print('----- 感度分析 -----')
print(f"1st order sobol' indices: {sobol_indices_1.round(4)}")
print(f"total sobol' indices: {sobol_indices_t.round(4)}")
