import math

import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt

train_x = torch.linspace(0, 1, 30).unsqueeze(1).double()
train_x = train_x[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
train_y = torch.sign(0.95+torch.cos(train_x * (4 * math.pi))).add(1).div(2).long().squeeze()  # 0 or 1
# train_y = torch.sign(torch.cos(train_x * (4 * math.pi))).long().squeeze()  # -1 or 1
# train_y = torch.sign(torch.cos(train_x * (4 * math.pi))).add(1).long().squeeze()  # 0 or 2

# train_x = torch.cat([train_x]*10, dim=0)
# train_y = torch.cat([train_y]*10, dim=0)

from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean, ZeroMean, LinearMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, ConstantKernel
from gpytorch.priors import LKJPrior, UniformPrior
from gpytorch.constraints import Interval


# We will use the simplest form of GP model, exact inference
class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        # self.mean_module = ZeroMean(batch_shape=torch.Size((num_classes,)))
        # self.mean_module = ConstantMean(
        #     batch_shape=torch.Size((num_classes,)),
        #     constant_constraint=Interval(0, 1),
        # )
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,))).double()

        # self.covar_module = ScaleKernel(
        #     RBFKernel(batch_shape=torch.Size((num_classes,))),
        #     batch_shape=torch.Size((num_classes,)),
        # ).double()
        self.covar_module = ScaleKernel(
            MaternKernel(batch_shape=torch.Size((num_classes,)), nu=1/2),
            batch_shape=torch.Size((num_classes,)),
        ).double()
        # self.covar_module = ScaleKernel(
        #     MaternKernel(batch_shape=torch.Size((num_classes,)), nu=3/2),
        #     batch_shape=torch.Size((num_classes,)),
        # ).double()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
# we let the DirichletClassificationLikelihood compute the targets for us
# likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True)
likelihood = DirichletClassificationLikelihood(train_y)
model = DirichletGPModel(train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)

# this is for running the notebook in our testing framework
import os
smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 50


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
# optimizer = torch.optim.Adam(model.parameters(), lr=1.)
# optimizer = torch.optim.Adam(model.parameters(), lr=10.)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, likelihood.transformed_targets).sum()
    loss.backward()
    # if i % 5 == 0:
    #     print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
    #         i + 1, training_iter, loss.item(),
    #         model.covar_module.base_kernel.lengthscale.mean().item(),
    #         model.likelihood.second_noise_covar.noise.mean().item()
    #     ))
    optimizer.step()


# Go into eval mode
model.eval()
likelihood.eval()

with torch.no_grad():
    # Test x are regularly spaced by 0.01 0,1 inclusive
    test_x = torch.linspace(0, 1, 1000).unsqueeze(1)

    # Get classification predictions numerically
    test_dist = model(test_x)
    pred_means = test_dist.loc

    # sampling from distribution
    pred_samples = test_dist.sample(torch.Size((256,))).exp()
    probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)
    # probabilities_std = (pred_samples / pred_samples.sum(-2, keepdim=True)).std(0)
    class_1_prob = probabilities[1]
    class_2_prob = probabilities[0]
    # class_1_std = probabilities_std[1]

    # Initialize fig and axes for plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')

    # Are these probabilities?
    ax.plot(test_x.numpy(), class_1_prob.float(), 'r')
    ax.plot(test_x.numpy(), class_2_prob.float(), 'g')
    # ax.plot(test_x.numpy(), class_1_prob.float() + class_1_std.float(), 'r')
    # ax.plot(test_x.numpy(), class_1_prob.float() - class_1_std.float(), 'r')

    # Get the predicted labels (probabilites of belonging to the positive class)
    # Transform these probabilities to be 0/1 labels
    pred_labels = class_1_prob.ge(0.5).float()
    ax.plot(test_x.numpy(), pred_labels.numpy(), 'b')
    ax.set_ylim([-1, 2])
    ax.legend(['Observed Data', 'Prob', 'Prob (not)', 'Predicted'])

    plt.show()
