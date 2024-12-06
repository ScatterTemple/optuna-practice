import skfda
import matplotlib.pyplot as plt

from skfda.representation.grid import FDataGrid
"""離散化された関数データを表します。
関数データを点のグリッドで離散化された曲線のセットとして表すクラス。
"""


# 0~18 歳の身長のデータ
X, y = skfda.datasets.fetch_growth(return_X_y=True)

X.plot()
plt.show()


# カナダの天気のデータ
X, y = skfda.datasets.fetch_weather(return_X_y=True)

X.plot()
plt.show()
