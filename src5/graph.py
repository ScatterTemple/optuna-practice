import numpy as np
import chaospy as cp
import plotly.graph_objects as go


def hist(x):
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=x,
        )
    )
    return fig


def _plot_scatter(x, y, mode):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode=mode,
        )
    )
    return fig


def plot(x, y):
    return _plot_scatter(x, y, 'lines')


def scatter(x, y):
    return _plot_scatter(x, y, 'markers')


def plot_dist(dist: cp.Distribution):
    x = dist.sample(size=np.array([100]), rule='grid', )
    y = dist.pdf(x)
    return plot(x, y)


if __name__ == '__main__':
    # x_ = np.array([0,1,2,3,4])
    # y_ = x_ ** 2
    # plot(x_, y_).show()

    dist = cp.Normal(0, 10)
    plot_dist(dist).show()

    def y(x):
        return x ** 2

    # hist(y(dist.sample(np.array([500]), rule='grid'))).show()
    hist(dist.sample(np.array([500]), rule='grid')).show()

    dist_mu1 = cp.Normal(1, 1)
    dist_mu1 ** 2
