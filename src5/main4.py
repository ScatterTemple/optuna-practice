"""Chaospy の代わりに力技で分散の計算を行います。"""

import random
import numpy
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.quasirandom import SobolEngine
from torch.distributions.normal import Normal
from SALib.sample import sobol as sample_sobol
from SALib.analyze import sobol as analyze_sobol
import plotly.graph_objects as go


def calc_VAR_SAMPLE_SIZE(upper, dim_):
    count = 0
    size = 0
    while size <= upper:
        count += 1
        # size = (2 * dim_ + 2) * count
        size = 2 ** count
    return size

# config
seed = None
dim = 10
TRAIING_SAMPLE_SIZE = 100 * dim
VAR_SAMPLE_SIZE = 50 * dim; VAR_SAMPLE_SIZE = calc_VAR_SAMPLE_SIZE(VAR_SAMPLE_SIZE, dim)
VAR_SAMPLE_METHOD = 'saltelli'
TEST_X_MEAN = 0.1
USE_FORCE_NO_NOISE = False

# seed
if seed is not None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# define problem
bound = [-1, 1]
bounds = [bound] * dim
def obj(x):
    return (x**2).sum(axis=1)


# training data
x = numpy.concat(
    [
        TEST_X_MEAN + numpy.zeros((1, dim)),
        numpy.random.rand(TRAIING_SAMPLE_SIZE, dim) * (bound[1] - bound[0]) + bound[0]
    ],
    axis=0
)
y = obj(x)


# training
def tensor(x_):
    return torch.tensor(numpy.array(x_), device='cpu', dtype=torch.double)

X = tensor(x)
Y = tensor(y)
if len(Y.shape) == 1:
    Y = Y.unsqueeze(0)
Y = Y.T
B = tensor(bounds).T
standardizer = Standardize(m=Y.shape[-1])
standardizer.forward(Y); _, YVar = standardizer.untransform(Y, 1e-6 * torch.ones_like(Y))
model = SingleTaskGP(
    train_X=X,
    train_Y=Y,
    train_Yvar=YVar,
    input_transform=Normalize(d=X.shape[-1], bounds=B),
    outcome_transform=standardizer,
)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)


# 確認
x_test = tensor([[TEST_X_MEAN]*dim])
post = model.posterior(x_test)
with torch.no_grad():
    print('model の精度確認')
    print('実際', obj(x_test).numpy())
    print('予測', post.mean.numpy()[:, 0])

# ===== 分散の伝搬の計算 =====

# ユーザー定義分散の計算
mean_list = numpy.array([TEST_X_MEAN] * dim)
sigma_list = numpy.array([0.1] * dim)

# saltelli を使う場合および感度分析に使う問題設定
problem = {
    'num_vars': dim,
    'names': [f'x{i}' for i in range(dim)],
    'bounds': [[m, s] for m, s in zip(mean_list, sigma_list)],  # When an entry for dists is not ‘unif’, the bounds entry does not indicate parameter bounds but sample-specific metadata.
    'dists': ['norm'] * dim
}

# ----- sobol を使わない場合...分散のばらつきが大きいので不採用 -----
if VAR_SAMPLE_METHOD == 'random':
    # 入力分布の計算
    covariance = torch.eye(dim, device='cpu', dtype=torch.double)
    for i, sigma in enumerate(sigma_list):
        covariance[i][i] = sigma**2
    x_dist = MultivariateNormal(
        loc=tensor(mean_list),
        covariance_matrix=covariance
    )

    # 入力分布からサンプルを作成
    x_dist_samples = x_dist.sample(
        torch.Size((VAR_SAMPLE_SIZE,))
    )

    # 確認
    # scatter(x_dist_samples[:, 0], x_dist_samples[:, 1]).show()
    # print(x_dist_samples[:, 0].var().sqrt())


# ----- sobol を使う場合 -----
elif VAR_SAMPLE_METHOD == 'sobol':
    # 入力分布の計算
    soboleng = SobolEngine(1, scramble=seed is not None)
    x_dist_list = [Normal(m, s) for m, s in zip(mean_list, sigma_list)]

    # 入力分布からサンプルを作成
    soboleng.draw(1)  # 最初は 0 で icdf が inf なので捨てる
    _tmp_samples = soboleng.draw(VAR_SAMPLE_SIZE, dtype=torch.double)
    uniform_samples_list = [_tmp_samples[torch.randperm(VAR_SAMPLE_SIZE)] for _ in range(dim)]
    x_dist_samples_list_per_dim = [x_dist.icdf(samples).flatten().numpy() for x_dist, samples in zip(x_dist_list, uniform_samples_list)]
    x_dist_samples = tensor(x_dist_samples_list_per_dim).T

    # 確認
    # scatter(x_dist_samples[:, 0], x_dist_samples[:, 1]).show()
    # print(x_dist_samples[:, 0].var().sqrt())

elif VAR_SAMPLE_METHOD == 'saltelli':
    # サンプルを作成
    # x_dist_samples = sobol_sample.sample(problem, N=int(VAR_SAMPLE_SIZE / (2 + 2 * dim)))
    x_dist_samples = sample_sobol.sample(problem, N=VAR_SAMPLE_SIZE)
    x_dist_samples = tensor(x_dist_samples)

else:
    raise NotImplementedError

# surrogate model を使ってマッピング
mapped_samples = model.posterior(x_dist_samples).mean.detach().numpy()

# 目的関数ごとに分散を計算
for i in range(mapped_samples.shape[-1]):
    mapped_samples_i: numpy.ndarray = mapped_samples[:, i]
    print('予測分散: ', mapped_samples_i.var().round(4))

    theoritical_v = (2 * sigma_list ** 2 * (sigma_list ** 2 + 2 * mean_list ** 2)).sum()
    print('非心カイ二乗分布の理論上の分散: ', float(numpy.array(theoritical_v).round(4)))


# 確認
# hist(mapped_samples_i).show()


# 感度分析
Si = analyze_sobol.analyze(
    problem,
    mapped_samples_i,
    print_to_console=True,
)
