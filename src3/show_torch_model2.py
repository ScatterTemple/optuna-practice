import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import unnormalize, normalize, standardize
# from botorch.utils import standardize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement

from botorch.optim import optimize_acqf

from gpytorch.constraints.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.means import ConstantMean, ZeroMean, LinearMean
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior


# helper function
def acqf_factory(ACQF_):

    class ModifiedACQF(ACQF_):
        model_c: SingleTaskGP

        def set_model(self, model):
            self.model_c = model

        def pof(self, X: torch.Tensor):
            # 予測点の平均と標準偏差をもとにした正規分布の関数を作る
            _X = X.squeeze(1)
            posterior = self.model_c.posterior(_X)
            mean = posterior.mean
            sigma = posterior.variance.sqrt()

            # 積分する
            normal = torch.distributions.Normal(mean, sigma)
            cdf = 1. - normal.cdf(torch.tensor(0., device='cpu').double())
            return cdf.squeeze(1)

        def forward(self, X: torch.Tensor):
            base_acqf = super().forward(X)
            pof = self.pof(X)

            # return torch.sigmoid(base_acqf) * pof
            return -torch.log(1 - torch.sigmoid(base_acqf)) * pof

    return ModifiedACQF


# helper function
def tensor(x: np.ndarray or dict):
    if isinstance(x, np.ndarray):
        return torch.Tensor(x, device='cpu').double()
    elif isinstance(x, dict):
        return tensor(np.array([data for data in x.values()]).T)
    else:
        return tensor(np.array(x))


# helper function
def symlog(x: np.ndarray or torch.Tensor):
    if isinstance(x, torch.Tensor):
        return torch.where(
            x >= 0.,
            torch.log(1. + x),
            -torch.log(1. - x),
        )
    else:
        ret = np.empty_like(x).astype(float)
        p_idx = np.where(x >= 0)
        n_idx = np.where(x < 0)
        ret[p_idx] = np.log(1. + x[p_idx])
        ret[n_idx] = -np.log(1. - x[n_idx])
        return ret


if __name__ == '__main__':

    # ===== problem =====
    print('''
    ===== 双曲線 xy > 0.25 の外にある円 =====
    
    最小化：(1/√2, 1/√2) からの距離
    拘束：xy > 0.25
    特徴：
        最適解：r=1, theta=0.79
        最悪解：r=1, theta=3.93
        特徴：r=0 のとき theta の値に関わらず obj=1
        特徴：theta=0.79 or 3.93 のとき r に比例
    
    ''')

    def get_next_point():
        x = prm['prm1'] * np.cos(prm['prm2'])
        y = prm['prm1'] * np.sin(prm['prm2'])

        # objective
        distance_from_ref = np.linalg.norm(
            np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]])  # reference point
            - np.array([x, y]),  # actual data point
            axis=0,
        )
        obj = dict(
            obj1=distance_from_ref,
        )

        # constraint
        c = np.abs(x) * np.abs(y)
        cns = dict(
            cns1=np.array([-1 if _c < 1./4. else 1 for _c in c]),
        )

        # ===== virtual trial =====
        for k in prm.keys():
            prm[k] = np.concatenate([prm[k]] * virtual_repeat, axis=0)
        for k in obj.keys():
            obj[k] = np.concatenate([obj[k]] * virtual_repeat, axis=0)
        for k in cns.keys():
            cns[k] = np.concatenate([cns[k]] * virtual_repeat, axis=0)

        # remove infeasible data
        prm_c = dict()
        obj_all = dict()
        idx = np.where(cns[tuple(cns.keys())[0]] == 1)
        for k in prm.keys():
            prm_c[k] = prm[k].copy()
            prm[k] = prm[k][idx]
        for k in obj.keys():
            obj_all[k] = obj[k].copy()
            obj[k] = obj[k][idx]

        # ===== train constraint =====
        train_X = normalize(tensor(prm_c), bounds=tensor(bounds))
        train_Y = tensor(cns)

        _batch_shape = SingleTaskGP.get_batch_dimensions(train_X, train_Y)[1]
        _ard_num_dims = train_X.shape[-1]
        gp_c = SingleTaskGP(
            train_X,
            train_Y,

            # これがないと未観測の点や疎な点が大多数に引っ張られるが、
            # これがあると pof 空間を虱潰ししにかかってしまう
            # でもこれがあっても虱潰しは発生する
            # train_Yvar=1e-4*torch.ones_like(train_Y),

            outcome_transform=Standardize(m=train_Y.shape[1],),

            mean_module=ZeroMean(batch_shape=_batch_shape),

            covar_module=ScaleKernel(
                base_kernel=MaternKernel(
                    nu=1/2,
                    ard_num_dims=_ard_num_dims,
                    batch_shape=_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                batch_shape=_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            ),

            # # pof を軽視しすぎる
            # covar_module=ScaleKernel(
            #     base_kernel=RBFKernel(
            #         ard_num_dims=_ard_num_dims,
            #         batch_shape=_batch_shape,
            #         # lengthscale_prior=GammaPrior(3.0, 6.0),
            #     ),
            #     batch_shape=_batch_shape,
            #     # outputscale_prior=GammaPrior(2.0, 0.15),
            # ),
        )
        mll_c = ExactMarginalLogLikelihood(gp_c.likelihood, gp_c)
        fit_gpytorch_mll(mll_c)

        # ===== train model =====
        train_X = normalize(tensor(prm), bounds=tensor(bounds))
        train_Y = -tensor(obj)  # maximize -> minimize

        gp = SingleTaskGP(
            train_X,
            train_Y,
            train_Yvar=1e-4*torch.ones_like(train_Y),
            outcome_transform=Standardize(m=train_Y.shape[1],)
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # ===== acqf =====
        ACQF = acqf_factory(LogExpectedImprovement)
        acqf = ACQF(
            model=gp,
            best_f=max(train_Y),
        )
        acqf.set_model(gp_c)

        # ===== optimize =====
        n_prms = train_X.shape[1]
        candidates, _acqf_values = optimize_acqf(
            acq_function=acqf,
            bounds=tensor([[0]*n_prms, [1]*n_prms]),
            q=(q := 1),
            num_restarts=10,
            raw_samples=512,
            sequential=True,
        )

        next_point = unnormalize(candidates[0], bounds=tensor(bounds)).detach().numpy()

        # ===== visualize =====
        if show_fig:
            # contour xy
            x = np.linspace(*bounds[tuple(prm.keys())[0]], 100)
            y = np.linspace(*bounds[tuple(prm.keys())[1]], 100)
            xx, yy = np.meshgrid(x, y)
            xy = tensor([xx.flatten(), yy.flatten()]).t()

            # helper function
            def show_contour(zz_, layout_):
                fig = go.Figure(layout=layout_)
                fig.add_trace(
                    go.Contour(
                        x=x,
                        y=y,
                        z=zz_,
                        colorscale='RdBu',
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=prm_c[tuple(prm_c.keys())[0]],
                        y=prm_c[tuple(prm_c.keys())[1]],
                        mode='markers',
                        marker=dict(color='black'),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=prm[tuple(prm.keys())[0]],
                        y=prm[tuple(prm.keys())[1]],
                        mode='markers',
                        marker=dict(color='green'),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[next_point[0]],
                        y=[next_point[1]],
                        mode='markers',
                        marker=dict(color='yellow', symbol='star', size=10, line_width=1),
                    )
                )
                fig.show()

            # predicted mean
            layout = go.Layout(title=f"Mean {count}")
            posterior = gp.posterior(normalize(xy, bounds=tensor(bounds)))
            zz = -posterior.mean.detach().numpy().flatten().reshape(xx.shape)  # maximize -> minimize
            show_contour(zz, layout)

            # predicted stddev
            layout = go.Layout(title=f"Stddev {count}")
            posterior = gp.posterior(normalize(xy, bounds=tensor(bounds)))
            zz = posterior.variance.sqrt().detach().numpy().flatten().reshape(xx.shape)
            show_contour(zz, layout)

            # acqf
            layout = go.Layout(title=f"ACQF {count}")
            acqf_values = acqf.forward(normalize(xy, bounds=tensor(bounds)).unsqueeze(q))
            zz = acqf_values.detach().numpy().flatten().reshape(xx.shape)
            if 'Log' in type(acqf).__name__:
                layout = go.Layout(title=f"ACQF (symlog) {count}")
                zz = symlog(zz)
            show_contour(zz, layout)

            # PoF
            layout = go.Layout(title=f"PoF {count}")
            posterior = gp_c.posterior(normalize(xy, bounds=tensor(bounds)))
            mean = posterior.mean
            std = posterior.variance.sqrt()
            zz = 1 - torch.distributions.Normal(mean, std).cdf(torch.tensor(0.)).detach().numpy().flatten().reshape(xx.shape)
            show_contour(zz, layout)

        for k in prm.keys():
            prm[k] = prm_c[k].copy()
        for k in obj.keys():
            obj[k] = obj_all[k]

        return next_point

    def update_problem():
        next_point_ = get_next_point()
        for k, nv in zip(prm.keys(), next_point_):
            prm[k] = np.concatenate([prm[k], [nv]], axis=0)

    # ===== main loop =====
    # helper function
    def is_optimized():
        x = prm[tuple(prm.keys())[0]]
        y = prm[tuple(prm.keys())[1]]
        coor = np.array([x, y]).T
        target = np.array([[1, np.pi / 4]])
        min_norm = np.linalg.norm(coor - target, axis=1).min()
        return min_norm < 0.1

    # ===== config =====
    n_startup_trials = 5
    n_samples = 20
    virtual_repeat = 1
    np.random.seed(43)
    show_fig = False

    import warnings

    warnings.filterwarnings('ignore', category=Warning)

    # 5 trials for reproductivity
    for _i in range(20):

        prm = dict()

        bounds = dict(
            prm1=[0, 1],
            prm2=[0, 2 * np.pi],
        )

        while True:

            for k in bounds.keys():
                lb, ub = bounds[k]
                new_value = np.random.rand() * (ub - lb) + lb
                if k not in prm:
                    prm[k] = np.array([new_value])
                else:
                    prm[k] = np.concatenate([prm[k], [new_value]], axis=0)

            # count feasible data
            x = prm['prm1'] * np.cos(prm['prm2'])
            y = prm['prm1'] * np.sin(prm['prm2'])
            c = np.abs(x) * np.abs(y)
            cns1 = [0 for _c in c if _c >= 1./4.]
            if len(cns1) >= n_startup_trials:
                break

        count = 0

        # main loop
        while not is_optimized():

            if _i == 0:
                show_fig = True
            else:
                show_fig = False

            update_problem()
            count += 1

            if _i == 0:
                input("一時停止中: enter to start")

            if count >= 20:
                break

        print(f'optimized with {count} trials.')
