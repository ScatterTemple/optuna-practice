import webbrowser

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, dcc, Output, Input
from dash.development.base_component import Component


class HeatmapApp:

    # ヒートマップ
    graph: dcc.Graph

    # 軸選択
    x_dropdown: Component
    y_dropdown: Component
    r_dropdown: Component
    g_dropdown: Component
    b_dropdown: Component

    def __init__(
            self,
            parameters: dict[str, np.ndarray],
            objectives: dict[str, np.ndarray],
            model: SingleTaskGP,
            bounds: dict[str, np.ndarray],
    ):
        self.app = Dash()
        self.p = parameters
        self.o = objectives
        self.model = model
        self.bounds = bounds



    def setup_components(self):
        self.graph = dcc.Graph()

        prm_names = list(self.p.keys())
        self.x_dropdown = dcc.Dropdown(prm_names, prm_names[0])
        self.y_dropdown = dcc.Dropdown(prm_names, prm_names[min(len(prm_names)-1, 1)])

        obj_names = list(self.o.keys())
        self.r_dropdown = dcc.Dropdown(obj_names, obj_names[0])
        self.g_dropdown = dcc.Dropdown(obj_names, obj_names[min(len(obj_names)-1, 1)])
        self.b_dropdown = dcc.Dropdown(obj_names, obj_names[min(len(obj_names)-1, 2)])

    def setup_layout(self):
        self.app.layout = [
            self.graph,
            html.Div(['x-axis', self.x_dropdown]),
            html.Div(['y-axis', self.y_dropdown]),
            html.Div(['color-r', self.r_dropdown]),
            html.Div(['color-g', self.g_dropdown]),
            html.Div(['color-b', self.b_dropdown]),
        ]

    def setup_callback(self):
        app = self.app

        @app.callback(
            output=Output(self.graph, 'figure'),
            inputs=dict(
                x_in=Input(self.x_dropdown, 'value'),
                y_in=Input(self.y_dropdown, 'value'),
                r_in=Input(self.r_dropdown, 'value'),
                g_in=Input(self.g_dropdown, 'value'),
                b_in=Input(self.b_dropdown, 'value'),
            ),
        )
        def update_graph(
                x_in, y_in, r_in, g_in, b_in,
        ):

            fig = go.Figure()

            used_x_values = self.p[x_in]
            used_y_values = self.p[y_in]
            used_r_values = self.o[r_in]
            used_g_values = self.o[g_in] if g_in is not None else [None]*len(used_r_values)
            used_b_values = self.o[b_in] if b_in is not None else [None]*len(used_r_values)

            x = np.sort(used_x_values)
            y = np.sort(used_y_values)
            r = np.array([[None]*len(used_x_values)]*len(used_y_values))
            g = np.array([[None]*len(used_x_values)]*len(used_y_values))
            b = np.array([[None]*len(used_x_values)]*len(used_y_values))

            for k, (_r, _g, _b) in enumerate(zip(used_r_values, used_g_values, used_b_values)):
                i = np.searchsorted(x, used_x_values[k])
                j = np.searchsorted(y, used_y_values[k])
                r[j, i] = _r
                g[j, i] = _g
                b[j, i] = _b

            opacity = 0.333

            fig.add_trace(
                go.Contour(
                    x=x,
                    y=y,
                    z=r,
                    # zsmooth='best',
                    opacity=opacity,
                    colorscale='reds',
                    connectgaps=True,
                ),

            )

            fig.add_trace(
                go.Contour(
                    x=x,
                    y=y,
                    z=g,
                    # zsmooth='best',
                    opacity=opacity,
                    colorscale='greens',
                    connectgaps=True,
                )
            )

            fig.add_trace(
                go.Contour(
                    x=x,
                    y=y,
                    z=b,
                    # zsmooth='best',
                    opacity=opacity,
                    colorscale='blues',
                    connectgaps=True,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=used_x_values,
                    y=used_y_values,
                    mode='markers',
                    marker=dict(color='black'),
                )
            )

            fig.update_layout(plot_bgcolor="white")
            fig.update_xaxes(linecolor='black', gridcolor='gray', mirror=True)
            fig.update_yaxes(linecolor='black', gridcolor='gray', mirror=True)

            return fig

    def run(self):
        self.setup_components()
        self.setup_callback()
        self.setup_layout()
        # webbrowser.open('http://localhost:8050')
        self.app.run(debug=True)


if __name__ == '__main__':

    h_app = HeatmapApp(
        dict(
            a=np.random.rand(100),
            b=np.random.rand(100),
            c=np.random.rand(100),
        ),
        dict(
            o=np.random.rand(100),
            c=np.random.rand(100),
        ),
    )
    h_app.run()
