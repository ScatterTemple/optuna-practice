import math
import torch
import gpytorch
from matplotlib import pyplot as plt

bounds = torch.Tensor([0, 1]).unsqueeze(1)
train_x = torch.linspace(0, 1, 20).unsqueeze(1).double()
train_y = torch.sign(0.9+torch.cos(train_x * (4 * math.pi))).add(1).div(2).double()  # 0 or 1

# train_x = torch.cat([train_x]*2, dim=0)
# train_y = torch.cat([train_y]*2, dim=0)

from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean, LinearMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, ConstantKernel
from gpytorch.priors import LKJPrior, UniformPrior
from gpytorch.constraints import Interval

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize, standardize

# initialize likelihood and model
# we let the DirichletClassificationLikelihood compute the targets for us

model = SingleTaskGP(
    normalize(train_x, bounds=bounds),
    train_y,
    1e-4*torch.ones_like(train_y),
    outcome_transform=Standardize(m=1),
)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)


with torch.no_grad():
    # Test x are regularly spaced by 0.01 0,1 inclusive
    test_x = torch.linspace(0, 1, 1000).unsqueeze(1)

    # Get classification predictions numerically
    posterior = model.posterior(test_x)
    mean = posterior.mean.squeeze()
    std = posterior.stddev

    # calc pof
    gamma = 1.
    pof = 1. - torch.distributions.Normal(mean, std).cdf(torch.tensor(0.5))

    # Initialize fig and axes for plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')

    # Are these probabilities?
    ax.plot(test_x.numpy(), mean.float(), 'r')
    ax.plot(test_x.numpy(), mean.float() + std.float(), 'r')
    ax.plot(test_x.numpy(), mean.float() - std.float(), 'r')

    # pof
    ax.plot(test_x.numpy(), pof.float(), 'g')

    # Get the predicted labels (probabilites of belonging to the positive class)
    # Transform these probabilities to be 0/1 labels
    pred_labels = mean.ge(0.5).float()
    ax.plot(test_x.numpy(), pred_labels.numpy(), 'b')
    ax.set_ylim([-1, 2])
    ax.legend(['Observed Data', 'Prob', 'Predicted'])

    plt.show()
