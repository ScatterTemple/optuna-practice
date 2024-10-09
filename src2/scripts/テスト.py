"""使い方の説明用ファイルです。"""

from src2.methods.random_manager import RandomManager
from src2.methods.botorch_manager import BayesianManager
from src2.problems.spiral import Spiral

from tqdm import tqdm


if __name__ == '__main__':
    problem = Spiral()
    random_manager = RandomManager(problem)
    bayesian_manager = BayesianManager(problem)

    for i in tqdm(range(100)):
        random_manager.sampling(False)

    # 出力する場合は True (表示)、パス（画像）を渡す
    random_manager.sampling(True, 'ランダム 100 回.png')

    # 別の手法で続きから実施する場合は
    # まずデータフレームを引き継ぐ
    bayesian_manager.df = random_manager.df

    bayesian_manager.sampling(True)
