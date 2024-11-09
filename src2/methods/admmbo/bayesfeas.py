import numpy as np
from src2.methods.admmbo.admmo_type import (
    Problem, F, C, ADMM, Opt,
    Input, Output, Inputs, Outputs, Bounds, ColVector, InputsEachCns
)
from src2.problems.base_problem import Floats
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=Warning)


import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor


# Define the main function
def bayesfeas(problem, opt, const_num):
    # Initialize
    samples = opt.c[const_num].initial_points
    C_values = np.zeros(samples.shape[0])
    h_values = np.zeros(samples.shape[0])

    C = problem.c
    h_func = lambda Z: (C[const_num](Z) > 0) + (opt.admm.rho / (2 * opt.admm.M)) * np.linalg.norm(opt.f.x - Z + (opt.f.y[const_num] / opt.admm.rho))**2
    CNST_func = lambda Z: C[const_num](Z)

    # Evaluate initial samples
    for i in range(samples.shape[0]):
        print(f'Running initial point #{i + 1}...')
        h_values[i] = h_func(samples[i, :])
        C_values[i] = CNST_func(samples[i, :])

    h_plus = np.min(h_values)

    meanfunc = opt.c[const_num].meanfunc
    covfunc = opt.c[const_num].covfunc
    hyp = {
        'lik': np.log(0.1),
        'cov': np.log([1 / 4, 1])  # ell = 1/4, sf = 1
    }

    # Main iteration loop
    for i in range(opt.c[const_num].max_iters):
        grid_data = np.zeros((opt.c[const_num].grid_size, opt.c[const_num].dims))

        # Random grid data within bounds
        for j in range(opt.c[const_num].dims):
            grid_data[:, j] = np.random.uniform(problem.bounds[j, 0], problem.bounds[j, 1], opt.c[const_num].grid_size)

        # Hypothetical GP hyperparameter optimization and prediction (assuming `scikit-learn`)
        # Note: Replace with actual GP library function calls as needed.
        gp = GaussianProcessRegressor()  # Example only; configure GP accordingly
        gp.fit(samples, C_values)
        m, s2 = gp.predict(grid_data, return_std=True)

        # EI calculation
        EI = EI_Z(m, s2, grid_data, h_plus, opt, const_num)

        if opt.c[const_num].optimize_ei:
            max_num = 10
            sortedEIValues = np.unique(np.sort(EI))

            if len(sortedEIValues) > max_num:
                maxEIValues = sortedEIValues[-max_num:]
            else:
                maxEIValues = sortedEIValues

            maxIndex = np.isin(EI, maxEIValues).nonzero()[0]

            Zstars = np.zeros((max_num, opt.c[const_num].dims))
            FinalMinEI_multi = np.zeros(max_num)

            for j in range(max_num):
                grid_sample = grid_data[maxIndex[j], :]
                Qz_val = Qz(grid_sample, h_plus, opt, const_num)

                # Build EI function
                if Qz_val < 0:
                    EIz_func = lambda z: 0
                elif Qz_val <= 1:
                    EIz_func = lambda z: -Qz(z, h_plus, opt, const_num) * norm.cdf(0, meanz(z, gp), s2z(z, gp))
                else:
                    EIz_func = lambda z: -Qz(z, h_plus, opt, const_num) * (1 - norm.cdf(0, meanz(z, gp), s2z(z, gp)))

                # Minimize -EI
                bounds = [(problem.bounds[k, 0], problem.bounds[k, 1]) for k in range(opt.c[const_num].dims)]
                res = minimize(EIz_func, grid_sample, bounds=bounds)
                Zstars[j, :] = res.x
                FinalMinEI_multi[j] = EIz_func(Zstars[j, :])

            FinalMinEI = np.min(FinalMinEI_multi)
            FinalMinEI_ind = np.argmin(FinalMinEI_multi)
            z_opt = Zstars[FinalMinEI_ind, :]
            EI_val = max(0, -FinalMinEI)
        else:
            FinalMaxEI_ind = np.argmax(EI)
            z_opt = grid_data[FinalMaxEI_ind, :]
            EI_val = EI[FinalMaxEI_ind]

        h_opt = h_func(z_opt)
        C_opt = CNST_func(z_opt)

        samples = np.vstack([samples, z_opt])
        h_values = np.append(h_values, h_opt)
        C_values = np.append(C_values, C_opt)

        if h_opt < h_plus:
            h_plus = h_opt

        print(f'Subproblem {const_num}, Iteration {i + len(opt.c[const_num].initial_points)}, ei = {EI_val}, value = {h_opt}, overall min = {h_plus}')

    h_min = np.min(h_values)
    min_Ind = np.argmin(h_values)
    zmin = samples[min_Ind, :]

    botrace = {
        'samples': samples,
        'values': C_values
    }

    return zmin, h_min, botrace


# Define helper functions
def Qz(Z, h_plus, opt, const_num):
    return h_plus - (opt.admm.rho / (2 * opt.admm.M)) * np.linalg.norm(opt.f.x - Z + (opt.f['y'][const_num] / opt.admm.rho))**2


def GPfinal(x, gp):
    mean_gp, var_gp = gp.predict([x], return_std=True)
    return mean_gp[0], var_gp[0]


def meanz(z, gp):
    return GPfinal(z, gp)[0]


def s2z(z, gp):
    return GPfinal(z, gp)[1]


def EI_Z(m, s2, grid_data, h_plus, opt, const_num):
    # Placeholder for Expected Improvement calculation
    return np.maximum(0, h_plus - m) / s2

