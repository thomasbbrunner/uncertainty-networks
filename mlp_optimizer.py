"""
This script is to test if there are differences between two methods of training
deep ensembles:
1) All models in the ensemble within one module, which is trained using a single
optimizer, and
2) Each model in the ensemble is a different module, which are trained using
different optimizers.

Result of this experiment:
No difference was noticed when both methods are trained using the 
*individual predictions* of each model.
It is worth noting that this comparison has to be done with different seed values,
as the performances depend strongly on the initial values of the parameters.
"""
from uncertainty_networks import UncertaintyMLP

import copy
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(3)

device = "cuda"
epochs = 1000

# generate dataset
X = torch.Tensor(np.linspace(-10, 10, 100)[..., None]).to(device)
Y = torch.sin(X).to(device)

def train(model, num_models):
    # repeat desired output to match shape of predictions
    Y_repeated = Y.unsqueeze(0).repeat_interleave(num_models, 0)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        optimizer.zero_grad()
        _, _, preds = model(X)
        preds = preds.flatten(0, 1)
        loss = loss_fn(preds, Y_repeated)
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == epochs - 1:
            print("Epoch: {}, Loss: {:.4f}".format(epoch, loss.item()))

    return model

def test_multiple(model):
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        model.eval()
        model.to(device)
        mean, _, preds = model(X)
        preds = preds.flatten(0, 1)
        loss = loss_fn(mean, Y)
        return (
            mean.to("cpu").numpy(),
            preds.to("cpu").numpy(),
            loss.to("cpu").numpy())

def test_single(models):
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        preds = torch.zeros((num_models, *X.shape), device=device)
        for i, model in enumerate(models):
            model.eval()
            model.to(device)
            _, _, preds[i] = model(X)
        mean = preds.mean(0)
        loss = loss_fn(mean, Y)
        return (
            mean.to("cpu").numpy(),
            preds.to("cpu").numpy(),
            loss.to("cpu").numpy())

num_models = 5

multiple = UncertaintyMLP(1, [64, 64], 1, torch.nn.LeakyReLU, None, 0, 1, num_models, device)
single = [UncertaintyMLP(1, [64, 64], 1, torch.nn.LeakyReLU, None, 0, 1, 1, device) for _ in range(num_models)]

trained_multiple = train(multiple, num_models)
trained_single = [train(single[i], 1) for i in range(num_models)]

multiple_mean, multiple_preds, multiple_loss = test_multiple(trained_multiple)
single_mean, single_preds, single_loss = test_single(trained_single)

fig = plt.figure(dpi=300, constrained_layout=True)
axs = fig.subplots(2, 2)
X_plot = X.to("cpu").numpy()
Y_plot = Y.to("cpu").numpy()

axs[0, 0].set_title("Single Model Containing Multiple Models\nTest Loss = {:.4}".format(multiple_loss), loc="left")
axs[0, 0].grid()
axs[0, 0].plot(X_plot, Y_plot, color="tab:blue", linestyle="--")
axs[0, 0].plot(X_plot, multiple_mean, color="tab:red")
axs[1, 0].set_title("Multiple Models\nTest Loss = {:.4}".format(single_loss), loc="left")
axs[1, 0].grid()
axs[1, 0].plot(X_plot, Y_plot, color="tab:blue", linestyle="--")
axs[1, 0].plot(X_plot, single_mean, color="tab:red")

axs[0, 1].set_title("Individual Predictions")
axs[0, 1].grid()
axs[1, 1].grid()
for pred in multiple_preds:
    axs[0, 1].plot(X_plot, pred, color="tab:red", alpha=0.2)
for pred in single_preds:
    axs[1, 1].plot(X_plot, pred, color="tab:red", alpha=0.2)

fig.savefig("mlp_optimizer.png")
