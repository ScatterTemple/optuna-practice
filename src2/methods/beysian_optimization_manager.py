from bayes_opt import BayesianOptimization


# 目的関数。(0, 1) で最大値 1 を取る。
def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1


# 境界条件をつける。
pbounds = {'x': (2, 4), 'y': (-3, 3)}

# optimizer の設定をする。
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
    # acquisition_function=None,    # このへんが必要そう
    # constraint=None,              # このへんが必要そう
    # bounds_transformer=None,      # 何これ
    # allow_duplicate_points=False, # 何これ
)

# 解きます。最大化がデフォルトみたい。
optimizer.maximize(
    init_points=2,  # Number of random points
    n_iter=3,       # Number of iterations (init_points 以外で)
)

# 解きました。
print(optimizer.max)

# 履歴も見られます。
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

# 問題の途中で境界条件を変更できます。
optimizer.set_bounds(new_bounds={"x": (-2, 3)})

# 追加で解きます。
# 前の評価で新しい bounds 違反の解があれば
# UserWarning が出ます。
optimizer.maximize(
    init_points=0,
    n_iter=5,
)

# 手動で評価する点を指定できます。
# 次に maximize() を呼ぶときに実行されます。
optimizer.probe(
    params={"x": 0.5, "y": 0.7},
    lazy=True,  # 即時実行したい場合は多分これを False にする
)

# ログをコンソールではなく json にしてみましょう
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
logger = JSONLogger(path="./logs.log")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
optimizer.maximize(
    init_points=2,
    n_iter=3,
)

# そうすると、読み込みができます。
from bayes_opt.util import load_logs
new_optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds={"x": (-2, 2), "y": (-2, 2)},
    verbose=2,
    random_state=7,
)
print(len(new_optimizer.space))

load_logs(new_optimizer, logs=["./logs.log"])

print("New optimizer is now aware of {} points.".format(len(new_optimizer.space)))

new_optimizer.maximize(
    init_points=0,
    n_iter=10,
)


# 獲得関数も指定できます。
from bayes_opt import acquisition

acq = acquisition.UpperConfidenceBound(kappa=2.5)

optimizer = BayesianOptimization(
    f=None,
    acquisition_function=acq,
    pbounds={'x': (-2, 2), 'y': (-3, 3)},
    verbose=2,
    random_state=1,
)

# maximize() は一気にやってしまうが、candidate だけでも動作させられる。
next_point_to_probe = optimizer.suggest()
print("Next point to probe is:", next_point_to_probe)
target = black_box_function(**next_point_to_probe)
print("Found the target value to be:", target)

# 評価結果を好きに optimizer に記録できる。
optimizer.register(
    params=next_point_to_probe,
    target=target,
)

# つまり maximize() が中でやってることはこういうことで、
# すきにカスタマイズできそう
for _ in range(5):
    next_point = optimizer.suggest()
    target = black_box_function(**next_point)
    optimizer.register(params=next_point, target=target)

    print(target, next_point)
print(optimizer.max)


# ===== 離散値を取ることは、原則できません =====
# 以下は無理やりの実装。

# 離散値を取る目的関数
def func_with_discrete_params(x, y, d):
    # Simulate necessity of having d being discrete.
    assert type(d) == int

    return ((x + y + d) // (1 + d)) / (1 + (x + y) ** 2)


# パラメータを離散値に変換して上記目的を計算する関数
def function_to_be_optimized(x, y, w):
    d = int(w)
    return func_with_discrete_params(x, y, d)


optimizer = BayesianOptimization(
    f=function_to_be_optimized,
    pbounds={'x': (-10, 10), 'y': (-10, 10), 'w': (0, 5)},
    verbose=2,
    random_state=1,
)

optimizer.set_gp_params(alpha=1e-3)
optimizer.maximize()


# callback を仕込むこともできる
# https://bayesian-optimization.github.io/BayesianOptimization/2.0.0/advanced-tour.html#Observers-Continued


# 複数の拘束を扱う例
import numpy as np
from scipy.optimize import NonlinearConstraint


def target_function(x, y):
    # Gardner is looking for the minimum, but this packages looks for maxima, thus the sign switch
    return np.cos(2*x)*np.cos(y) + np.sin(x)


def constraint_function_2_dim(x, y):
    return np.array([
        - np.cos(x) * np.cos(y) + np.sin(x) * np.sin(y),
        - np.cos(x) * np.cos(-y) + np.sin(x) * np.sin(-y)])


constraint_lower = np.array([-np.inf, -np.inf])
constraint_upper = np.array([0.6, 0.6])

constraint = NonlinearConstraint(constraint_function_2_dim, constraint_lower, constraint_upper)
optimizer = BayesianOptimization(
    f=target_function,
    constraint=constraint,
    pbounds=pbounds,
    verbose=0, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(
    init_points=3,
    n_iter=10,
)
