import numpy
import chaospy


# chaospy の seed は numpy のそれに従う
numpy.random.seed(1234)

# 入力
x = numpy.linspace(0, 10, 1000)


# 何らかのモデル。このサンプルスクリプトでは入力は固定。
def model_solver(parameters, coordinates=x):
    """Simple ordinary differential equation solver."""
    alpha, beta = parameters
    return alpha*numpy.e**(-coordinates*beta)


# 入寮を確率分布にできる
alpha = chaospy.Normal(1.5, 0.2)  # 正規分布
beta = chaospy.Uniform(0.1, 0.2)  # 一様分布
joint = chaospy.J(alpha, beta)  # ?

# 分布を polynomial 表現に展開することもできる
from packaging.version import Version
if Version("1.20.0") <= Version(numpy.__version__) < Version("2.0.0"):
    raise Exception('Use numpy 2.0.0 or later.')
else:

    # noinspection PyTypeChecker
    # これは分布の多項式表現
    expansion: chaospy.ndpoly = chaospy.generate_expansion(
        8,  # Order of polynomial expansion.
        joint
    )

    # 交互作用含め 5 次の項までの表示
    # noinspection PyUnresolvedReferences
    print(expansion[:5].round(8))

# 確率分布 joint から sobol サンプリング
# noinspection PyTypeChecker
parameter_samples: numpy.ndarray = joint.sample(1000, rule="sobol")  # 2 * 1000 ndarray
parameter_samples[:, :4].round(8)

# サンプリングされた parameter を実行
# (x はグローバルスコープで定義)
evaluations: numpy.ndarray = numpy.array([model_solver(sample) for sample in parameter_samples.T])

# noinspection PyTypeChecker
# 入力パラメータ分布の多項式表現、入力パラメータ(X)、出力(Y) から多項式近似モデルを作る
# これは入力パラメータの分布を加味した元のモデルの多項式近似モデル
approx_solver: chaospy.ndpoly = chaospy.fit_regression(expansion, parameter_samples, evaluations)

# 多項式近似モデルに対し実際の入力パラメータの分布を入れて
# 出力の期待値を計算できる（このサンプルでは入力 x は固定なので
# 出力 y のパラメータ依存性がわかる）
expected: numpy.ndarray = chaospy.E(approx_solver, joint)  # (1000,) ndarray


# 同様に stddev も計算できる
deviation: numpy.ndarray = chaospy.Std(approx_solver, joint)  # (1000,) ndarray

# この stddev が、入力ばらつきに対する出力のばらつきといっていい。
# しかし、 joint を入力したせいで個々の変数の影響とかは不明になる。

# 一次感度分析（Sobol'指標の計算...joint は平均と分散を有するので、これでいい？）
# 定義は、分散の比（VarXi / VarY）らしい。
sobol_indices: numpy.ndarray = chaospy.Sens_m(approx_solver, joint)  # 2 * 1000 ndarray
# 定義からして、以下の値はだいたい一緒になる。（モデルに交互作用がほぼない？ため）
print(deviation[:5])
print((sobol_indices[0] * 0.2)[:5])

# 二次感度分析
# 多分対角成分のみに意味がある？
# サンプルが単純だから パラメータ**2 に影響する効果がないだけ？
sobol_indices_m2: numpy.ndarray = chaospy.Sens_m2(approx_solver, joint)  # (2, 2, 1000) ndarray
print(sobol_indices_m2[:, :, :5])

# 総感度指数 よくわかんない。二次以降の感度指標も総計した数字らしい。
# ただし交互作用項は両方に入ってくるから合計が 1 にならないらしい。
sobol_indices_t: numpy.ndarray = chaospy.Sens_t(approx_solver, joint)  # (2, 2, 1000) ndarray
print(sobol_indices_t[:, :5])  # (2, 1000) ndarray
