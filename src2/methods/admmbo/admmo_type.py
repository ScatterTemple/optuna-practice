import numpy as np
Input = list[float] or np.ndarray
Output = list[float] or np.ndarray
Inputs = list[Input] or np.ndarray
Outputs = list[Output] or np.ndarray
InputsEachCns = list[Inputs] or np.ndarray
OutputsEachCns = list[Outputs] or np.ndarray
Bounds = list[list[float, float]] or np.ndarray
ColVector = list[list[float]] or np.ndarray


class Problem(object):
    f: str  # function name
    f_parameters: list  # additional arguments
    c: list[str]  # function names
    c_parameters: dict[str, list]  # additional arguments
    bounds: Bounds  # bounds for input
    InfPoint: Input  # very wrong point?
    F: callable  # = lambda x: fun(x, *args)  # 未知の目的
    C: list[callable]  # = lambda x: fun(x, *args)  # 未知の拘束


# ## Experimantal setup
# rho=.1
# M=50
# initial_bud_f=10
# final_bud=5
# InfPoint=[-1000 -1000]
# initial_num = 2
# num=5
# y=0
# tolC = 0.01


class F(object):
    grid_size: int  # = 10^5  # Size of grid to select candidate hyperparameters from.(?)
    max_iters: int  # = []  # Maximum number of function evaluations.
    step_iters: int  # = initial_bud_f
    reduced_step_iters: int  # = final_bud
    meanfunc: list[callable]  # = [meanShiftedBar:=lambda x: NotImplemented]  # Constant mean function.
    covfunc: list[callable, float]  # = [covMaterniso:=lambda x: NotImplemented, 5]
    hyp: float  # = -1 # Set hyperparameters using MLE.
    dims: int  # = dim  # Number of parameters.
    mins: Input  # = problem.bounds(:,1)' # Minimum value for each of the parameters. Should be 1-by-opt.dims
    maxes: Input  # = problem.bounds(:,2)'
    save_trace: bool  # = false
    y: InputsEachCns
    z: InputsEachCns
    # for j=1:length(problem.c)
    #     y{j}=ones(1,dim).*y
    #     z{j}=bounds(:,2)'
    rho: float  # = rho

    initial_points: Inputs


class C(object):
    grid_size: int  # = 10^5;
    max_iters: int  # =[]; % Maximum number of function evaluations.
    step_iters: float  # =initial_bud_c;
    reduced_step_iters: float  # =final_bud;%
    meanfunc: list[callable]  # = {@meanZero}; % Constant mean function.
    covfunc: list[callable, float]  # ={@covMaterniso, 5};
    dims: int  # =2;
    optimize_ei: bool  # = false;

    initial_points: Inputs


class ADMM(object):
    max_iters: int  #
    rho: float  # .1
    M: float  # 50


class Opt(object):
    f: F
    c: list[C]
    ADMM: ADMM
    _n_obj: int
